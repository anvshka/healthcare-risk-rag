"""Session-local services for the Stage 6 Streamlit interface; no model fitting."""
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import pandas as pd
from .data_loading import load_tables
from .explain import RiskExplainer, load_selected_bundle
from .features import AdmissionFeatureBuilder
from .prolonged_stay import EXCLUDED_FEATURE, load_secondary_bundle
from .rag import (GroundedAssistant, KnowledgeIndex, SentenceEmbedder, digest,
                  read_sources, validate_shap_record)


def project_signature(root: Path) -> str:
    """Fingerprint inputs so a rerun cannot retain stale model/source resources."""
    paths = []
    for folder in ['data/raw', 'models', 'data/knowledge']:
        directory = root / folder
        paths.extend(p for p in directory.glob('*') if p.is_file() and p.name != '.gitkeep')
    for name in ['data/processed/stage5/rag_explanations.json',
                 'data/processed/stage5/index/manifest.json']:
        path = root / name
        if path.is_file():
            paths.append(path)
    signature = hashlib.sha256(str(root.resolve()).encode())
    for path in sorted(paths):
        signature.update(str(path.relative_to(root)).encode())
        signature.update(path.read_bytes())
    return signature.hexdigest()


def shap_hash(record: dict[str, Any]) -> str:
    """Link a generated narrative to the exact current attribution record."""
    return digest(json.dumps(record, sort_keys=True, allow_nan=False))


def cached_rag_is_current(result: dict[str, Any], record: dict[str, Any],
                          fingerprint: str, source_hashes: dict[str, str],
                          provider: str) -> bool:
    """Require matching attribution, knowledge, identity and provider provenance."""
    return (result.get('shap_record_hash') == shap_hash(record)
            and result.get('index_fingerprint') == fingerprint
            and str(result.get('hadm_id')) == str(record['hadm_id'])
            and str(result.get('subject_id')) == str(record['subject_id'])
            and result.get('provider') == provider
            and bool(result.get('disclaimer'))
            and all(source_hashes.get(s.get('filename')) == s.get('content_hash')
                    for s in result.get('sources', [])))


class ProjectSession:
    """Keep model and retrieval resources within one Streamlit user's session.

    No raw-data uploads, automatic embedding downloads, fitting or publishing.
    Current model SHAP records are computed locally, even when a saved narrative
    is reused. The UI recreates this object whenever project inputs change.
    """
    def __init__(self, root: Path) -> None:
        self.root = root
        os.environ.setdefault('MPLCONFIGDIR', str((root / 'reports/.matplotlib').resolve()))
        self.bundle = load_selected_bundle(root / 'models')
        secondary_path = root / 'models/prolonged_stay_ensemble.pkl'
        self.secondary_bundle = load_secondary_bundle(secondary_path) if secondary_path.is_file() else None
        self.raw = load_tables(root / 'data/raw')
        self.explainer = RiskExplainer(self.bundle, self.raw)
        self.catalog = self.explainer.metadata.copy()
        self.catalog['scope'] = ['training' if flag else 'heldout' for flag in self.explainer.training_mask]
        self._records: dict[str, dict[str, Any]] = {}
        self._contexts: dict[tuple[str, str], dict[str, Any]] = {}
        self._assistant: GroundedAssistant | None = None
        self._provider: str | None = None

    def prolonged_stay_score(self, hadm_id: str) -> float | None:
        """Return the frozen soft-voting score for the separate secondary outcome."""
        if self.secondary_bundle is None:
            return None
        key = str(hadm_id).strip().upper()
        matches = self.catalog.index[self.catalog['hadm_id'].astype(str).eq(key)].tolist()
        if len(matches) != 1:
            raise ValueError('Admission ID was not found uniquely')
        builder = AdmissionFeatureBuilder.from_state(self.secondary_bundle['feature_state'])
        features = builder.transform(self.raw).drop(columns=[EXCLUDED_FEATURE])
        encoded = self.secondary_bundle['encoder'].transform(features.iloc[[matches[0]]])
        scores = [model.predict_proba(encoded)[:, 1][0] for model in self.secondary_bundle['models'].values()]
        return float(sum(scores) / len(scores))

    def admissions(self, scope: str = 'heldout') -> pd.DataFrame:
        """Return the current patient/admission catalog without target labels."""
        if scope not in {'heldout', 'all'}:
            raise ValueError('Admission scope must be heldout or all')
        return (self.catalog if scope == 'all' else self.catalog.loc[self.catalog.scope.eq('heldout')]).copy()

    def model_record(self, hadm_id: str) -> dict[str, Any]:
        """Calculate and cache frozen-model SHAP for an unambiguous admission ID."""
        key = str(hadm_id).strip().upper()
        if key not in self._records:
            self._records[key] = self.explainer.explain_admission(key)
        return self._records[key]

    def _saved_context(self, record: dict[str, Any], provider: str) -> dict[str, Any] | None:
        """Reuse Stage 5 output only if the whole current knowledge set matches."""
        output = self.root / 'data/processed/stage5'
        try:
            sources = read_sources(self.root / 'data/knowledge')
            hashes = {source.filename: source.content_hash for source in sources}
            manifest = json.loads((output / 'index/manifest.json').read_text(encoding='utf-8'))
            original = {source['filename']: source['content_hash'] for source in manifest['config']['sources']}
            if original != hashes:
                return None
            saved = json.loads((output / 'rag_explanations.json').read_text(encoding='utf-8'))
            for result in saved:
                if cached_rag_is_current(result, record, manifest['fingerprint'], hashes, provider):
                    return result
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            return None
        return None

    def context(self, record: dict[str, Any], provider: str = 'none') -> dict[str, Any]:
        """Use verified saved context, or retrieve locally with cached model weights."""
        validate_shap_record(record)
        key = (shap_hash(record), provider)
        if key in self._contexts:
            return self._contexts[key]
        saved = self._saved_context(record, provider)
        if saved is not None:
            self._contexts[key] = saved
            return saved
        if self._assistant is None or self._provider != provider:
            # Deliberately local-only: a UI visit must not download a model.
            embedder = SentenceEmbedder(self.root / 'data/cache/embeddings', allow_download=False)
            index = KnowledgeIndex.build(self.root / 'data/knowledge',
                self.root / 'data/processed/stage5/index', embedder)
            self._assistant = GroundedAssistant(index, provider=provider)
            self._provider = provider
        result = self._assistant.explain(record)
        self._contexts[key] = result
        return result
