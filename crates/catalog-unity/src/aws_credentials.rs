//! AWS S3 credential provider backed by Unity Catalog.
//!
//! Unity Catalog vends temporary S3 credentials that expire (~1 hour). The
//! object store caches the value we return and asks again on each request, so
//! we cache it here and re-vend from Unity Catalog once the token nears its
//! `expiration_time`. This keeps a long-lived store (e.g. a table being
//! followed) working past the initial token's lifetime.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use deltalake_core::logstore::object_store::aws::AwsCredential;
use deltalake_core::logstore::object_store::{
    CredentialProvider, Error as ObjectStoreError, Result as ObjectStoreResult,
};
use tokio::sync::Mutex;

use crate::models::TableTempCredentialsResponse;
use crate::{UnityCatalog, UnityCatalogError};

/// Re-vend this long before the token actually expires, so we never hand out a
/// credential that lapses mid-request.
const EXPIRY_SKEW: Duration = Duration::seconds(60);

/// An [`object_store::CredentialProvider`] that vends AWS S3 credentials from
/// Unity Catalog and refreshes them before they expire.
pub struct UnityS3CredentialProvider {
    catalog: Arc<UnityCatalog>,
    catalog_id: String,
    database_name: String,
    table_name: String,
    cache: Mutex<Option<CachedCredential>>,
}

struct CachedCredential {
    credential: Arc<AwsCredential>,
    expires_at: DateTime<Utc>,
}

impl std::fmt::Debug for UnityS3CredentialProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the cached credential.
        f.debug_struct("UnityS3CredentialProvider")
            .field("catalog_id", &self.catalog_id)
            .field("database_name", &self.database_name)
            .field("table_name", &self.table_name)
            .finish()
    }
}

impl UnityS3CredentialProvider {
    pub fn new(
        catalog: Arc<UnityCatalog>,
        catalog_id: impl Into<String>,
        database_name: impl Into<String>,
        table_name: impl Into<String>,
    ) -> Self {
        Self {
            catalog,
            catalog_id: catalog_id.into(),
            database_name: database_name.into(),
            table_name: table_name.into(),
            cache: Mutex::new(None),
        }
    }

    /// Vend fresh temporary credentials from Unity Catalog, preferring
    /// read/write and falling back to read-only (mirrors
    /// `get_uc_location_and_token`).
    async fn vend(&self) -> Result<CachedCredential, UnityCatalogError> {
        let response = self
            .catalog
            .get_temp_table_credentials_with_permission(
                &self.catalog_id,
                &self.database_name,
                &self.table_name,
                "READ_WRITE",
            )
            .await?;

        let temp = match response {
            TableTempCredentialsResponse::Success(temp) => temp,
            TableTempCredentialsResponse::Error(rw_error) => {
                match self
                    .catalog
                    .get_temp_table_credentials(
                        &self.catalog_id,
                        &self.database_name,
                        &self.table_name,
                    )
                    .await?
                {
                    TableTempCredentialsResponse::Success(temp) => temp,
                    TableTempCredentialsResponse::Error(read_error) => {
                        return Err(UnityCatalogError::TemporaryCredentialsFetchFailure {
                            error_code: read_error.error_code,
                            message: format!(
                                "READ_WRITE failed: {}. READ failed: {}",
                                rw_error.message, read_error.message
                            ),
                        });
                    }
                }
            }
        };

        let expires_at = temp.expiration_time;
        let aws = temp
            .aws_temp_credentials
            .ok_or(UnityCatalogError::MissingCredential)?;

        Ok(CachedCredential {
            credential: Arc::new(AwsCredential {
                key_id: aws.access_key_id,
                secret_key: aws.secret_access_key,
                token: aws.session_token,
            }),
            expires_at,
        })
    }
}

#[async_trait]
impl CredentialProvider for UnityS3CredentialProvider {
    type Credential = AwsCredential;

    async fn get_credential(&self) -> ObjectStoreResult<Arc<AwsCredential>> {
        let mut guard = self.cache.lock().await;

        if let Some(cached) = guard.as_ref()
            && cached.expires_at - EXPIRY_SKEW > Utc::now()
        {
            return Ok(Arc::clone(&cached.credential));
        }

        let fresh = self.vend().await.map_err(|e| ObjectStoreError::Generic {
            store: "UnityCatalog",
            source: Box::new(e),
        })?;
        let credential = Arc::clone(&fresh.credential);
        *guard = Some(fresh);
        Ok(credential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UnityCatalogBuilder;
    use crate::client::ClientOptions;
    use crate::models::tests::GET_TABLE_RESPONSE;
    use httpmock::prelude::*;

    const TABLE_PATH: &str = "/api/2.1/unity-catalog/tables/catalog_name.schema_name.table_name";
    const CREDS_PATH: &str = "/api/2.1/unity-catalog/temporary-table-credentials";

    fn temp_creds_body(expires_at: DateTime<Utc>) -> String {
        format!(
            r#"{{"aws_temp_credentials":{{"access_key_id":"test_key","secret_access_key":"test_secret","session_token":"test_token"}},"expiration_time":{},"url":"s3://bucket/table"}}"#,
            expires_at.timestamp_millis()
        )
    }

    async fn provider_for(server: &MockServer) -> UnityS3CredentialProvider {
        let options = ClientOptions::builder().allow_http(true).build();
        let catalog = UnityCatalogBuilder::builder()
            .workspace_url(server.url(""))
            .bearer_token("bearer_token")
            .client_options(options)
            .build()
            .build()
            .unwrap();
        UnityS3CredentialProvider::new(Arc::new(catalog), "catalog_name", "schema_name", "table_name")
    }

    #[tokio::test]
    async fn caches_credential_until_near_expiry() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.path(TABLE_PATH).method("GET");
                then.body(GET_TABLE_RESPONSE);
            })
            .await;
        let creds_mock = server
            .mock_async(|when, then| {
                when.path(CREDS_PATH).method("POST");
                then.body(temp_creds_body(Utc::now() + Duration::hours(1)));
            })
            .await;

        let provider = provider_for(&server).await;
        let first = provider.get_credential().await.unwrap();
        let second = provider.get_credential().await.unwrap();

        assert_eq!(first.key_id, "test_key");
        assert_eq!(first.token.as_deref(), Some("test_token"));
        assert_eq!(first.key_id, second.key_id);
        // A valid token is cached: two reads hit Unity Catalog only once.
        assert_eq!(creds_mock.hits_async().await, 1);
    }

    #[tokio::test]
    async fn revends_when_expired() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.path(TABLE_PATH).method("GET");
                then.body(GET_TABLE_RESPONSE);
            })
            .await;
        // Token is already within the refresh skew, so every read must re-vend.
        let creds_mock = server
            .mock_async(|when, then| {
                when.path(CREDS_PATH).method("POST");
                then.body(temp_creds_body(Utc::now()));
            })
            .await;

        let provider = provider_for(&server).await;
        provider.get_credential().await.unwrap();
        provider.get_credential().await.unwrap();

        // Two reads of an expired token re-vend twice; a provider that cached
        // forever (the bug) would show only 1 hit.
        assert_eq!(creds_mock.hits_async().await, 2);
    }
}
