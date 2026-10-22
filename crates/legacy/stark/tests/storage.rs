#![cfg(not(target_family = "wasm"))]

mod common;

use std::{sync::Arc, time::Duration};

use download_manager::{DestinationLock, LockOwner};
use kiban::rt::RuntimeHandle;
use mock_registry::{Behavior, MockRegistry, artifact_path};
use rstest::rstest;
use stark::{
    engine::Downloader,
    registry::mirai::HUGGING_FACE_URL,
    storage::{DownloadManagerType, DownloadPhase, DownloadState, Storage},
    types::{
        basic::{File, Hash, HashMethod, Repository},
        model::{Model, ModelAccessibility, ModelIdentifier, ModelSource},
    },
};
use tokio::time::timeout;
use tokio_stream::{StreamExt, wrappers::BroadcastStream};
use uuid::Uuid;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, header_exists, method, path},
};

use crate::common::TestStorage;

const HELLO: &[u8] = b"hello\n";
const HELLO_SHA256: &str = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03";
const HELLO_GIT_BLOB_SHA1: &str = "ce013625030ba8dba906f756967f9e9ca394464a";
const REVISION: &str = "f5dec40ceb1d8c4b15f049bf5e185dd7be2cc150";

fn synthetic_registry_model(
    file_name: String,
    version: &str,
) -> Model {
    Model::external(
        "synthetic/model".into(),
        "test".into(),
        "Test".into(),
        "uzu".into(),
        "Uzu".into(),
        "1".into(),
        vec![],
        ModelAccessibility::OnDevice {
            source: ModelSource::Registry {
                toolchain_version: version.into(),
                repository: None,
                source_repository: None,
                files: vec![File {
                    url: "http://127.0.0.1:9/model".into(),
                    name: file_name,
                    size: 1,
                    hashes: vec![Hash {
                        method: HashMethod::Sha256,
                        value: HELLO_SHA256.into(),
                    }],
                }],
            },
        },
        None,
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn registry_refresh_rejects_escaping_paths_without_deleting_the_victim() -> Result<(), Box<dyn std::error::Error>>
{
    let test_storage =
        TestStorage::new(RuntimeHandle::current(), vec![], DownloadManagerType::Universal, HUGGING_FACE_URL, None)
            .await?;
    let victim_directory = tempfile::tempdir()?;
    let victim = victim_directory.path().join("victim");
    std::fs::write(&victim, b"synthetic-victim")?;
    for (name, version) in [
        (victim.to_string_lossy().to_string(), "1"),
        ("../../../../victim".into(), "1"),
        ("model.safetensors".into(), "../../outside"),
        ("model.safetensors".into(), "/outside"),
        ("weights.lock".into(), "1"),
        ("weights.PART".into(), "1"),
    ] {
        let model = synthetic_registry_model(name, version);
        assert!(test_storage.storage.refresh(&[model], true).await.is_err());
        assert_eq!(std::fs::read(&victim)?, b"synthetic-victim");
    }
    let valid = synthetic_registry_model("nested/model.safetensors".into(), "1");
    test_storage.storage.refresh(&[valid.clone()], false).await?;
    assert!(test_storage.storage.cache_model_path(&valid).is_some());
    Ok(())
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn registry_refresh_rejects_symlinked_destinations_and_sidecars() -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::fs::symlink;
    let test_storage =
        TestStorage::new(RuntimeHandle::current(), vec![], DownloadManagerType::Universal, HUGGING_FACE_URL, None)
            .await?;
    let outside = tempfile::tempdir()?;
    let victim = outside.path().join("victim");
    std::fs::write(&victim, b"synthetic-victim")?;
    let model = synthetic_registry_model("model.safetensors".into(), "1");
    let cache = test_storage.storage.cache_model_path(&model).unwrap();
    std::fs::create_dir_all(&cache)?;
    for name in [
        "model.safetensors",
        "model.safetensors.part",
        "model.safetensors.lock",
        "model.safetensors.checksum",
        "model.safetensors.resume_data",
    ] {
        symlink(&victim, cache.join(name))?;
        assert!(test_storage.storage.refresh(&[model.clone()], true).await.is_err());
        assert_eq!(std::fs::read(&victim)?, b"synthetic-victim");
        std::fs::remove_file(cache.join(name))?;
    }
    symlink(outside.path(), cache.join("nested"))?;
    let nested = synthetic_registry_model("nested/victim".into(), "1");
    assert!(test_storage.storage.refresh(&[nested], false).await.is_err());
    assert_eq!(std::fs::read(&victim)?, b"synthetic-victim");
    Ok(())
}

#[rstest]
#[case::universal(DownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::native(DownloadManagerType::Native))]
#[tokio::test(flavor = "multi_thread")]
async fn model_lifecycle(#[case] kind: DownloadManagerType) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let model = registry.models.first().ok_or("mock registry must include a model")?;
    let test_storage =
        TestStorage::new(RuntimeHandle::current(), vec![model.clone()], kind, HUGGING_FACE_URL, None).await?;
    let storage = &test_storage.storage;
    let identifier = model.identifier.clone();
    let cache_path = storage.cache_model_path(model).ok_or("model must have a cache path")?;
    let mut events = storage.subscribe();
    let total_bytes = storage.state(&identifier).await?.total_bytes;

    storage.download(&identifier).await?;
    let downloading = wait_for(storage, &identifier, &mut events, |state| {
        state.total_bytes > 0 && state.downloaded_bytes >= state.total_bytes / 4
    })
    .await;
    assert!(matches!(downloading.phase, DownloadPhase::Downloading {}), "got {:?}", downloading.phase);

    storage.pause(&identifier).await?;
    let paused =
        wait_for(storage, &identifier, &mut events, |state| matches!(state.phase, DownloadPhase::Paused {})).await;
    assert!(paused.downloaded_bytes >= total_bytes / 4);

    storage.download(&identifier).await?;
    let downloaded =
        wait_for(storage, &identifier, &mut events, |state| matches!(state.phase, DownloadPhase::Downloaded {})).await;
    assert_eq!(downloaded.downloaded_bytes, total_bytes);
    for served in registry.files.iter() {
        let destination = cache_path.join(&served.file.name);
        assert_eq!(tokio::fs::read(&destination).await?, served.bytes.to_vec());
        assert!(artifact_path(&destination, "checksum").is_file());
    }

    storage.delete(&identifier).await?;
    let deleted = storage.state(&identifier).await?;
    assert!(matches!(deleted.phase, DownloadPhase::NotDownloaded {}));
    assert_eq!(deleted.downloaded_bytes, 0);
    for served in registry.files.iter() {
        let destination = cache_path.join(&served.file.name);
        assert!(!destination.exists());
        assert!(!artifact_path(&destination, "checksum").exists());
    }

    let served = registry.files.first().ok_or("mock registry must include files")?;
    let destination = cache_path.join(&served.file.name);
    tokio::fs::create_dir_all(&cache_path).await?;
    tokio::fs::write(&destination, served.bytes.as_ref()).await?;
    let _lock = DestinationLock::acquire(
        &destination,
        &LockOwner {
            manager_id: "foreign-manager".to_string(),
            instance_id: Uuid::new_v4(),
        },
    )
    .await?;
    let refused = storage.delete(&identifier).await.expect_err("delete must be refused while locked");
    assert!(refused.to_string().contains("foreign-manager"), "unexpected error: {refused}");
    assert!(destination.exists());

    let mut changed = model.clone();
    if let ModelAccessibility::OnDevice {
        source: ModelSource::Registry {
            files,
            ..
        },
    } = &mut changed.accessibility
    {
        files.truncate(1);
    }
    storage.refresh(&[changed.clone()], false).await?;
    assert_eq!(storage.state(&identifier).await?.total_bytes, served.file.size);
    storage.refresh(&[model.clone(), changed.clone()], false).await?;
    assert_eq!(storage.state(&identifier).await?.total_bytes, total_bytes);
    storage.refresh(&[changed, model.clone()], false).await?;
    assert_eq!(storage.state(&identifier).await?.total_bytes, served.file.size);
    Ok(())
}

#[rstest]
#[case::universal(DownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::native(DownloadManagerType::Native))]
#[tokio::test(flavor = "multi_thread")]
async fn downloader_streams(#[case] kind: DownloadManagerType) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let model = registry.models.first().ok_or("mock registry must include a model")?;
    let test_storage =
        TestStorage::new(RuntimeHandle::current(), vec![model.clone()], kind, HUGGING_FACE_URL, None).await?;
    let storage = Arc::clone(&test_storage.storage);
    let identifier = model.identifier.clone();
    let downloader = Downloader::new(identifier.clone(), Arc::clone(&storage));
    let mut events = storage.subscribe();

    downloader.resume().await?;
    let progress = downloader.progress().await?;
    wait_for(&storage, &identifier, &mut events, |state| matches!(state.phase, DownloadPhase::Downloading {})).await;
    downloader.pause().await?;
    timeout(Duration::from_secs(10), async { while progress.next().await.is_some() {} }).await?;
    let paused =
        wait_for(&storage, &identifier, &mut events, |state| matches!(state.phase, DownloadPhase::Paused {})).await;
    assert!(!paused.is_in_progress());
    assert!(!paused.can_pause());

    downloader.resume().await?;
    let progress = downloader.progress().await?;
    let last = timeout(Duration::from_secs(30), async {
        let mut last = None;
        while let Some(state) = progress.next().await {
            last = Some(state);
        }
        last
    })
    .await?;
    assert!(matches!(last.map(|state| state.phase), Some(DownloadPhase::Downloaded {})));
    assert!(downloader.progress().await?.next().await.is_none());
    Ok(())
}

#[rstest]
#[case::universal(DownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::native(DownloadManagerType::Native))]
#[tokio::test(flavor = "multi_thread")]
async fn hugging_face_model(#[case] kind: DownloadManagerType) -> Result<(), Box<dyn std::error::Error>> {
    let hugging_face = MockServer::start().await;
    let route = format!("/trymirai/model/resolve/{REVISION}/model.safetensors");
    Mock::given(method("GET"))
        .and(path(route.clone()))
        .and(header("authorization", "Bearer hf_test"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(HELLO))
        .mount(&hugging_face)
        .await;
    // Like the Mirai CDN, this origin rejects any bearer token, so its file must download anonymously.
    let cdn = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/config.json"))
        .and(header_exists("authorization"))
        .respond_with(ResponseTemplate::new(401))
        .with_priority(1)
        .mount(&cdn)
        .await;
    Mock::given(method("GET"))
        .and(path("/config.json"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(HELLO))
        .mount(&cdn)
        .await;
    let files = vec![
        File {
            url: format!("{}/config.json", cdn.uri()),
            name: "config.json".to_string(),
            size: HELLO.len() as i64,
            hashes: vec![Hash {
                method: HashMethod::GitBlobSha1,
                value: HELLO_GIT_BLOB_SHA1.to_string(),
            }],
        },
        File {
            url: format!("{}{route}", hugging_face.uri()),
            name: "model.safetensors".to_string(),
            size: HELLO.len() as i64,
            hashes: vec![Hash {
                method: HashMethod::Sha256,
                value: HELLO_SHA256.to_string(),
            }],
        },
    ];
    let model = Model::external(
        "pinned".to_string(),
        "mirai".to_string(),
        "Mirai".to_string(),
        "uzu".to_string(),
        "Uzu".to_string(),
        "1".to_string(),
        vec![],
        ModelAccessibility::OnDevice {
            source: ModelSource::Registry {
                toolchain_version: "1".to_string(),
                repository: Some(Repository {
                    identifier: "trymirai/model".to_string(),
                    commit_hash: Some(REVISION.to_string()),
                    paths: None,
                }),
                source_repository: None,
                files,
            },
        },
        None,
    );
    let test_storage =
        TestStorage::new(RuntimeHandle::current(), vec![model.clone()], kind, &hugging_face.uri(), Some("hf_test"))
            .await?;
    let storage = &test_storage.storage;
    let cache_path = storage.cache_model_path(&model).ok_or("model must have a cache path")?;
    assert_eq!(cache_path.file_name().and_then(|name| name.to_str()), Some(REVISION));
    let mut events = storage.subscribe();

    storage.download(&model.identifier).await?;
    let state = wait_for(storage, &model.identifier, &mut events, |state| {
        matches!(state.phase, DownloadPhase::Downloaded {} | DownloadPhase::Error { .. })
    })
    .await;
    assert!(matches!(state.phase, DownloadPhase::Downloaded {}), "got {:?}", state.phase);
    for name in ["config.json", "model.safetensors"] {
        let destination = cache_path.join(name);
        assert_eq!(tokio::fs::read(&destination).await?, HELLO);
        assert!(artifact_path(&destination, "checksum").is_file());
    }
    Ok(())
}

async fn wait_for(
    storage: &Storage,
    identifier: &ModelIdentifier,
    events: &mut BroadcastStream<(ModelIdentifier, DownloadState)>,
    mut is_expected: impl FnMut(&DownloadState) -> bool,
) -> DownloadState {
    timeout(Duration::from_secs(30), async {
        let state = storage.state(identifier).await.expect("model state must exist");
        if is_expected(&state) {
            return state;
        }
        while let Some(result) = events.next().await {
            let (event_identifier, state) = result.expect("storage event stream must not lag");
            if &event_identifier == identifier && is_expected(&state) {
                return state;
            }
        }
        panic!("storage event stream ended before the expected state");
    })
    .await
    .expect("timed out waiting for storage state")
}
