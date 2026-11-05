//! JSON-RPC client of the wallet (spec/node.md section 7.2): one HTTP/1.1 POST
//! per call, each on a new TCP connection.
//!
//! The node answers every request with status 200 and a `Content-Length` body
//! whose `error` member carries the failure, then closes the connection. The
//! client therefore reads to end of stream and never inspects the status line;
//! chunked bodies and keep-alive are not supported.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use neb_json::Value;

#[derive(Clone, Debug)]
pub struct RpcClient {
    url: String,
    host: String,
}

impl RpcClient {
    /// Accepts `http://127.0.0.1:8432` or `127.0.0.1:8432`; a path or another
    /// scheme is an error.
    pub fn new(url: impl Into<String>) -> Result<RpcClient, String> {
        let url = url.into();
        let host = url.strip_prefix("http://").unwrap_or(&url).trim_end_matches('/').to_string();
        if host.is_empty() || host.contains('/') {
            return Err("rpc url must be http://host:port".into());
        }
        Ok(RpcClient { url, host })
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    /// Returns the `result` member. A JSON-RPC error becomes `"message (code N)"`.
    /// Reads and writes time out after 30 s; the connect has no timeout of its
    /// own.
    pub fn call(&self, method: &str, params: Vec<Value>) -> Result<Value, String> {
        let mut request = BTreeMap::new();
        request.insert("jsonrpc".to_string(), Value::String("1.0".into()));
        request.insert("id".to_string(), Value::Number(1));
        request.insert("method".to_string(), Value::String(method.into()));
        request.insert("params".to_string(), Value::Array(params));
        let body = Value::Object(request).to_string();
        let mut stream = TcpStream::connect(&self.host).map_err(|e| format!("connect {}: {e}", self.host))?;
        let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
        let head = format!(
            "POST / HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.host,
            body.len()
        );
        stream.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
        stream.write_all(body.as_bytes()).map_err(|e| e.to_string())?;
        stream.flush().map_err(|e| e.to_string())?;
        let mut response = Vec::new();
        stream.read_to_end(&mut response).map_err(|e| e.to_string())?;
        let text = String::from_utf8(response).map_err(|_| "response is not utf8".to_string())?;
        let (_, body) = text.split_once("\r\n\r\n").ok_or("response has no body")?;
        let value = neb_json::parse(body).map_err(|e| e.to_string())?;
        if let Some(error) = value.get("error")
            && !error.is_null()
        {
            let message = error.get("message").and_then(Value::as_str).unwrap_or("rpc error");
            let code = error.get("code").and_then(Value::as_i64).unwrap_or(0);
            return Err(format!("{message} (code {code})"));
        }
        value.get("result").cloned().ok_or_else(|| "response has no result".to_string())
    }

    pub fn block_count(&self) -> Result<u64, String> {
        self.call("getblockcount", vec![])?.as_u64().ok_or_else(|| "block count is not an integer".to_string())
    }

    pub fn block_hash(&self, height: u32) -> Result<String, String> {
        self.call("getblockhash", vec![Value::Number(height as i64)])?
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| "block hash is not a string".to_string())
    }

    pub fn block_hex(&self, hash: &str) -> Result<String, String> {
        self.call("getblock", vec![Value::String(hash.to_string())])?
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| "block is not a string".to_string())
    }

    pub fn send_raw_transaction(&self, hex: &str) -> Result<String, String> {
        self.call("sendrawtransaction", vec![Value::String(hex.to_string())])?
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| "transaction id is not a string".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The URL reduces to host:port; a path is refused.
    #[test]
    fn url_parsing() {
        let c = RpcClient::new("http://127.0.0.1:8332").unwrap();
        assert_eq!(c.host, "127.0.0.1:8332");
        assert!(RpcClient::new("http://127.0.0.1:8332/path").is_err());
    }
}
