use std::net::IpAddr;

use rocket::{Request, http::Status};

/// A configured key authorizes use of the single operator-owned model.
/// The environment keeps the key out of command-line arguments and logs.
pub struct ServerAccess {
    key: Option<String>,
    loopback: bool,
}

impl ServerAccess {
    pub fn new(
        address: IpAddr,
        key: Option<String>,
        allow_unauthenticated_local: bool,
    ) -> Result<Self, &'static str> {
        if key.as_ref().is_some_and(|key| key.is_empty() || !key.bytes().all(|byte| byte.is_ascii_graphic())) {
            return Err("STARK_SERVER_API_KEY must be nonempty printable ASCII");
        }
        if key.is_none() && !(address.is_loopback() && allow_unauthenticated_local) {
            return Err("set STARK_SERVER_API_KEY or explicitly allow unauthenticated loopback access");
        }
        Ok(Self {
            key,
            loopback: address.is_loopback(),
        })
    }

    pub fn authorize(
        &self,
        request: &Request<'_>,
    ) -> Result<(), Status> {
        // This native-client API has no browser credential or CORS workflow.
        if request.headers().get_one("Origin").is_some() {
            return Err(Status::Forbidden);
        }
        if let Some(key) = &self.key {
            let supplied = request.headers().get_one("Authorization").and_then(|value| value.strip_prefix("Bearer "));
            if !supplied.is_some_and(|supplied| equal_secret(supplied.as_bytes(), key.as_bytes())) {
                return Err(Status::Unauthorized);
            }
        } else if self.loopback {
            let host = request.headers().get_one("Host").unwrap_or_default();
            let hostname = host
                .strip_prefix('[')
                .and_then(|host| host.split_once(']').map(|(host, _)| host))
                .unwrap_or_else(|| host.split(':').next().unwrap_or_default());
            if hostname != "localhost" && !hostname.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback()) {
                return Err(Status::Forbidden);
            }
        }
        Ok(())
    }
}

fn equal_secret(
    left: &[u8],
    right: &[u8],
) -> bool {
    let mut difference = left.len() ^ right.len();
    for (index, byte) in right.iter().enumerate() {
        difference |= usize::from(left.get(index).copied().unwrap_or_default() ^ byte);
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use rocket::{get, http::Header, local::asynchronous::Client, routes};

    use super::*;

    #[get("/chat/completions")]
    fn protected(_info: &super::super::request_info::RequestInfo) -> &'static str {
        "ok"
    }

    #[rocket::async_test]
    async fn both_aliases_require_the_configured_key_and_reject_browser_origins() {
        let access = ServerAccess::new("127.0.0.1".parse().unwrap(), Some("synthetic-key".into()), false).unwrap();
        let client = Client::tracked(
            rocket::build().manage(access).mount("/", routes![protected]).mount("/v1", routes![protected]),
        )
        .await
        .unwrap();
        for path in ["/chat/completions", "/v1/chat/completions"] {
            assert_eq!(client.get(path).dispatch().await.status(), Status::Unauthorized);
            assert_eq!(
                client.get(path).header(Header::new("Authorization", "Bearer wrong")).dispatch().await.status(),
                Status::Unauthorized
            );
            assert_eq!(
                client.get(path).header(Header::new("Authorization", "Bearer synthetic-key")).dispatch().await.status(),
                Status::Ok
            );
            assert_eq!(
                client
                    .get(path)
                    .header(Header::new("Authorization", "Bearer synthetic-key"))
                    .header(Header::new("Origin", "https://example.invalid"))
                    .dispatch()
                    .await
                    .status(),
                Status::Forbidden
            );
        }
    }

    #[test]
    fn public_binding_cannot_opt_out_of_authentication() {
        assert!(ServerAccess::new("0.0.0.0".parse().unwrap(), None, true).is_err());
        assert!(ServerAccess::new("127.0.0.1".parse().unwrap(), None, false).is_err());
        assert!(ServerAccess::new("127.0.0.1".parse().unwrap(), None, true).is_ok());
    }
}
