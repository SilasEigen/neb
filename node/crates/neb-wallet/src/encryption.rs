//! Password encryption of the key file.
//!
//! An encrypted key file is UTF-8 text:
//!
//! ```text
//! # neb wallet, encrypted with a password
//! neb-wallet-encrypted 1
//! salt <32 hex>
//! nonce <48 hex>
//! data <hex>
//! ```
//!
//! `data` is the plain key file (one 64-hex secret per line) sealed with XChaCha20-Poly1305. The 32-byte key comes
//! from the password with Argon2id (64 MiB, 3 passes, 1 lane) and the 16-byte random salt; the nonce is 24 random
//! bytes, and the first line after the comment, the salt and the nonce are authenticated with the data. A wrong
//! password and a modified file give the same error.

use argon2::{Algorithm, Argon2, Params, Version};
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use rand_core::{OsRng, RngCore};

const MAGIC: &str = "neb-wallet-encrypted 1";
const KDF_MEMORY_KIB: u32 = 64 * 1024;
const KDF_PASSES: u32 = 3;
const KDF_LANES: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncryptionError {
    /// The password is wrong or the file was modified.
    Password,
    /// The file is not an encrypted key file of this version.
    Format(String),
}

impl std::fmt::Display for EncryptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncryptionError::Password => write!(f, "wrong password, or the wallet file was modified"),
            EncryptionError::Format(e) => write!(f, "encrypted wallet file: {e}"),
        }
    }
}

impl std::error::Error for EncryptionError {}

/// True when `text` is an encrypted key file.
pub fn is_encrypted(text: &str) -> bool {
    text.lines().map(str::trim).any(|l| l == MAGIC)
}

fn derive_key(password: &str, salt: &[u8]) -> [u8; 32] {
    let params = Params::new(KDF_MEMORY_KIB, KDF_PASSES, KDF_LANES, Some(32)).expect("valid Argon2 parameters");
    let mut key = [0u8; 32];
    Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
        .hash_password_into(password.as_bytes(), salt, &mut key)
        .expect("a 16-byte salt and a 32-byte output are accepted");
    key
}

fn associated_data(salt: &[u8], nonce: &[u8]) -> Vec<u8> {
    [MAGIC.as_bytes(), salt, nonce].concat()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str, what: &str) -> Result<Vec<u8>, EncryptionError> {
    if !text.len().is_multiple_of(2) {
        return Err(EncryptionError::Format(format!("{what} is not hex")));
    }
    (0..text.len() / 2)
        .map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16))
        .collect::<Result<_, _>>()
        .map_err(|_| EncryptionError::Format(format!("{what} is not hex")))
}

/// Seals the plain key file `plain` with `password` and returns the text of the encrypted file.
pub fn encrypt(plain: &str, password: &str) -> String {
    let mut salt = [0u8; 16];
    let mut nonce = [0u8; 24];
    OsRng.fill_bytes(&mut salt);
    OsRng.fill_bytes(&mut nonce);
    let cipher = XChaCha20Poly1305::new(&derive_key(password, &salt).into());
    let aad = associated_data(&salt, &nonce);
    let data = cipher
        .encrypt(&XNonce::from(nonce), Payload { msg: plain.as_bytes(), aad: &aad })
        .expect("XChaCha20-Poly1305 seals any message of this size");
    format!(
        "# neb wallet, encrypted with a password\n{MAGIC}\nsalt {}\nnonce {}\ndata {}\n",
        hex(&salt),
        hex(&nonce),
        hex(&data)
    )
}

/// Opens an encrypted key file and returns the plain key file.
pub fn decrypt(text: &str, password: &str) -> Result<String, EncryptionError> {
    let field = |name: &str| {
        text.lines()
            .find_map(|l| l.trim().strip_prefix(name).and_then(|rest| rest.strip_prefix(' ')))
            .ok_or_else(|| EncryptionError::Format(format!("no {name} line")))
    };
    if !is_encrypted(text) {
        return Err(EncryptionError::Format(format!("no \"{MAGIC}\" line")));
    }
    let salt = unhex(field("salt")?, "salt")?;
    let nonce = unhex(field("nonce")?, "nonce")?;
    let data = unhex(field("data")?, "data")?;
    if salt.len() != 16 || nonce.len() != 24 {
        return Err(EncryptionError::Format("the salt is 16 bytes and the nonce 24".into()));
    }
    let cipher = XChaCha20Poly1305::new(&derive_key(password, &salt).into());
    let aad = associated_data(&salt, &nonce);
    let plain = cipher
        .decrypt(
            &XNonce::try_from(nonce.as_slice()).expect("24 bytes, checked above"),
            Payload { msg: &data, aad: &aad },
        )
        .map_err(|_| EncryptionError::Password)?;
    String::from_utf8(plain).map_err(|_| EncryptionError::Password)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAIN: &str = "# neb testnet wallet: one 64-hex secret per line\n0101010101010101010101010101010101010101010101010101010101010101\n";

    // The plain text comes back with the right password, and the secret never appears in the file.
    #[test]
    fn a_sealed_file_opens_with_its_password_only() {
        let sealed = encrypt(PLAIN, "correct horse");
        assert!(is_encrypted(&sealed));
        assert!(!sealed.contains("0101010101"));
        assert_eq!(decrypt(&sealed, "correct horse").unwrap(), PLAIN);
        assert_eq!(decrypt(&sealed, "wrong horse"), Err(EncryptionError::Password));
    }

    // Two seals of the same text differ (fresh salt and nonce), and a changed byte of the data is refused.
    #[test]
    fn seals_are_fresh_and_tamper_evident() {
        let a = encrypt(PLAIN, "pw");
        assert_ne!(a, encrypt(PLAIN, "pw"));
        let data_at = a.find("data ").unwrap() + 5;
        let mut bytes = a.into_bytes();
        bytes[data_at] = if bytes[data_at] == b'0' { b'1' } else { b'0' };
        assert_eq!(decrypt(&String::from_utf8(bytes).unwrap(), "pw"), Err(EncryptionError::Password));
    }

    #[test]
    fn a_plain_file_is_not_encrypted() {
        assert!(!is_encrypted(PLAIN));
        assert!(matches!(decrypt(PLAIN, "pw"), Err(EncryptionError::Format(_))));
    }
}
