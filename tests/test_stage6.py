"""Stage 6 service/UI tests use isolated unit fixtures, not substitute datasets."""
import copy
import json
from pathlib import Path
import pickle
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from test_stage2 import tables
from test_stage3 import fitted,metadata
from test_stage5 import FixtureEmbedder,write_source
from src.app_service import ProjectSession,cached_rag_is_current,project_signature,shap_hash
from src.rag import GroundedAssistant,KnowledgeIndex

APP_FILE=Path(__file__).resolve().parents[1]/'app/app.py'


@pytest.fixture
def app_project(tmp_path,tables,fitted,monkeypatch):
    root=tmp_path/'project'
    raw=root/'data/raw'
    raw.mkdir(parents=True)
    for name,frame in tables.items():
        frame.to_csv(raw/f'{name}.csv',index=False)
    models=root/'models'
    models.mkdir()
    bundle={'version':1,'model_name':'random_forest',
        'pipeline':fitted.pipelines['random_forest'],'feature_state':fitted.builder.state(),
        'encoded_feature_names':fitted.encoded_feature_names}
    (models/'selected_model.json').write_text(json.dumps({'selected_model':'random_forest','bundle':'random_forest.pkl'}))
    with (models/'random_forest.pkl').open('wb') as stream:
        pickle.dump(bundle,stream)
    knowledge=root/'data/knowledge'
    knowledge.mkdir(parents=True)
    write_source(knowledge,'demo_PLACEHOLDER.md','placeholder','PLACEHOLDER. Retrieval test only, not medical evidence.')
    index=KnowledgeIndex.build(knowledge,root/'data/processed/stage5/index',FixtureEmbedder())
    session=ProjectSession(root)
    assistant=GroundedAssistant(index,provider='none',min_score=-1)
    records=[assistant.explain(session.model_record(hadm)) for hadm in session.catalog.hadm_id]
    (root/'data/processed/stage5/rag_explanations.json').write_text(json.dumps(records))
    monkeypatch.setenv('HEALTHCARE_PROJECT_ROOT',str(root))
    monkeypatch.setenv('LLM_PROVIDER','none')
    return root


def test_catalog_contains_correct_scope_and_no_outcomes(app_project):
    session=ProjectSession(app_project)
    assert session.admissions().hadm_id.tolist()==['C1']
    assert len(session.admissions('all'))==4
    assert 'hospital_expire_flag' not in session.catalog
    with pytest.raises(ValueError):
        session.admissions('unsupported')


def test_record_reuses_frozen_model_state(app_project):
    session=ProjectSession(app_project)
    before=json.dumps(session.explainer.builder.state(),sort_keys=True)
    first=session.model_record('c1')
    second=session.model_record('C1')
    assert first is second
    assert first['scope']=='heldout'
    assert json.dumps(session.explainer.builder.state(),sort_keys=True)==before
    assert first['reconstruction_error']<1e-6


def test_saved_context_avoids_model_download(app_project,monkeypatch):
    import src.app_service as module
    def forbidden(*args,**kwargs):
        pytest.fail('Valid saved narratives must not load embedding weights')
    monkeypatch.setattr(module,'SentenceEmbedder',forbidden)
    session=ProjectSession(app_project)
    record=session.model_record('C1')
    result=session.context(record)
    assert result['knowledge_status']=='placeholder_only'
    assert result['shap_record_hash']==shap_hash(record)


def test_provenance_rejects_score_identity_provider_and_source_changes(app_project):
    session=ProjectSession(app_project)
    record=session.model_record('C1')
    result=session.context(record)
    manifest=json.loads((app_project/'data/processed/stage5/index/manifest.json').read_text())
    hashes={source['filename']:source['content_hash'] for source in manifest['config']['sources']}
    assert cached_rag_is_current(result,record,manifest['fingerprint'],hashes,'none')
    altered=copy.deepcopy(record)
    altered['probability']=0.9
    assert not cached_rag_is_current(result,altered,manifest['fingerprint'],hashes,'none')
    assert not cached_rag_is_current(result,record,'different',hashes,'none')
    assert not cached_rag_is_current(result,record,manifest['fingerprint'],hashes,'http')
    assert not cached_rag_is_current(result,record,manifest['fingerprint'],{},'none')


def test_source_edit_invalidates_saved_context_and_project_signature(app_project):
    session=ProjectSession(app_project)
    record=session.model_record('C1')
    assert session._saved_context(record,'none') is not None
    before=project_signature(app_project)
    path=app_project/'data/knowledge/demo_PLACEHOLDER.md'
    path.write_text(path.read_text()+'\nChanged test context.')
    assert project_signature(app_project)!=before
    assert session._saved_context(record,'none') is None


def test_live_context_is_local_only(app_project,monkeypatch):
    import src.app_service as module
    captured=[]
    def local_embedder(cache_dir,allow_download):
        captured.append(allow_download)
        return FixtureEmbedder()
    monkeypatch.setattr(module,'SentenceEmbedder',local_embedder)
    path=app_project/'data/processed/stage5/rag_explanations.json'
    path.unlink()
    session=ProjectSession(app_project)
    result=session.context(session.model_record('C1'))
    assert captured==[False]
    assert result['provider']=='none'
    assert result['knowledge_status']=='placeholder_only'


def test_unknown_admission_rejected(app_project):
    with pytest.raises(ValueError,match='Unknown'):
        ProjectSession(app_project).model_record('missing')


def test_ui_default_selection_and_score(app_project):
    at=AppTest.from_file(str(APP_FILE),default_timeout=40).run()
    assert not at.exception and not at.error
    assert at.selectbox(key='subject_id').value=='C'
    assert at.selectbox(key='hadm_id').value=='C1'
    session=ProjectSession(app_project)
    expected=session.model_record('C1')['probability']
    assert at.metric[0].value==f'{expected:.1%}'
    assert any('Placeholder knowledge only' in w.value for w in at.warning)
    assert len(at.dataframe[0].value)==5
    assert any('not medical advice' in c.value.lower() for c in at.caption)


def test_ui_patient_admission_switch_and_training_warning(app_project):
    at=AppTest.from_file(str(APP_FILE),default_timeout=40).run()
    at.radio(key='scope').set_value('All admissions').run()
    at.selectbox(key='subject_id').set_value('A').run()
    assert at.selectbox(key='hadm_id').options==['A1','A2']
    at.selectbox(key='hadm_id').set_value('A2').run()
    assert not at.exception and not at.error
    assert any('not an independent test prediction' in w.value for w in at.warning)
    assert any(s.value=='Admission A2' for s in at.subheader)
    at.radio(key='scope').set_value('Held-out admissions').run()
    assert at.selectbox(key='subject_id').value=='C'
    assert at.selectbox(key='hadm_id').value=='C1'
    assert not at.exception


def test_ui_missing_project_gives_setup_steps(tmp_path,monkeypatch):
    monkeypatch.setenv('HEALTHCARE_PROJECT_ROOT',str(tmp_path/'missing'))
    at=AppTest.from_file(str(APP_FILE),default_timeout=40).run()
    assert not at.exception
    assert any('not ready' in e.value for e in at.error)
    assert not at.metric
    assert any('src.train' in code.value for code in at.code)


def test_ui_missing_retrieval_does_not_hide_score(app_project,monkeypatch):
    import src.app_service as module
    (app_project/'data/processed/stage5/rag_explanations.json').unlink()
    def missing(*args,**kwargs):
        raise ValueError('Embedding model unavailable. Run python -m src.rag --download-model')
    monkeypatch.setattr(module,'SentenceEmbedder',missing)
    at=AppTest.from_file(str(APP_FILE),default_timeout=40).run()
    assert not at.exception
    assert len(at.metric)==3
    assert any('cited context is not ready' in warning.value for warning in at.warning)
    assert any('src.rag --download-model' in code.value for code in at.code)


def test_config_is_loopback_and_no_telemetry():
    text=(APP_FILE.parents[1]/'.streamlit/config.toml').read_text()
    assert 'address = "127.0.0.1"' in text
    assert 'gatherUsageStats = false' in text

def test_ui_punctuation_is_not_mojibake(app_project):
    at=AppTest.from_file(str(APP_FILE),default_timeout=40).run()
    captions=' '.join(c.value for c in at.caption)
    assert chr(0xC2)+chr(0xB7) not in captions
    assert chr(0xE2)+chr(0x20AC) not in captions
    assert not at.exception
