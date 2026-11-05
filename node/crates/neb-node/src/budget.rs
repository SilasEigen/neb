//! What one peer may make the node spend (spec section 8.1): a budget of
//! ticket evaluations per second for each session (one for the proof of a
//! block, one for each of its shares), and a ban on the addresses that send a
//! block with an invalid proof of work or share.

use std::collections::BTreeMap;
use std::net::IpAddr;
use std::num::NonZeroU32;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Default ticket evaluations per second for one peer session (`--pow-budget`).
pub const DEFAULT_POW_BUDGET: u32 = 200;
/// Duration of a ban for sending a block with an invalid proof of work or share.
pub const BAN_TIME: Duration = Duration::from_secs(24 * 60 * 60);

/// One token, counted in millionths so that the arithmetic stays exact.
const TOKEN: u64 = 1_000_000;

/// A token bucket that refills `rate` tokens per second and holds at most
/// `rate` tokens, so a session may burst for one second.
///
/// Overflow: `tokens <= rate * TOKEN < 2^32 * 10^6 < 2^52` and the refill term
/// is at most `10^6 * rate < 2^52`, so the sum fits in `u64`.
pub struct TokenBucket {
    rate: u64,
    tokens: u64,
    last: Instant,
}

impl TokenBucket {
    pub fn new(rate: NonZeroU32, now: Instant) -> TokenBucket {
        let rate = u64::from(rate.get());
        TokenBucket { rate, tokens: rate * TOKEN, last: now }
    }

    /// Takes one token, or returns how long until one is available. The refill
    /// is capped at one second of elapsed time, which equals the bucket size.
    pub fn take_at(&mut self, now: Instant) -> Result<(), Duration> {
        let elapsed = now.saturating_duration_since(self.last).as_micros().min(1_000_000) as u64;
        self.last = self.last.max(now);
        self.tokens = (self.tokens + elapsed * self.rate).min(self.rate * TOKEN);
        if self.tokens >= TOKEN {
            self.tokens -= TOKEN;
            Ok(())
        } else {
            Err(Duration::from_micros((TOKEN - self.tokens).div_ceil(self.rate)))
        }
    }

    /// Blocks the calling thread until a token is taken.
    pub fn wait(&mut self) {
        while let Err(delay) = self.take_at(Instant::now()) {
            std::thread::sleep(delay);
        }
    }
}

/// Banned peer addresses with the time each ban ends. The list lives in
/// memory only. Expired entries are dropped when a new ban is added.
#[derive(Default)]
pub struct BanList {
    until: Mutex<BTreeMap<IpAddr, Instant>>,
}

impl BanList {
    pub fn ban(&self, ip: IpAddr) {
        self.ban_at(ip, Instant::now());
    }

    pub fn is_banned(&self, ip: IpAddr) -> bool {
        self.is_banned_at(ip, Instant::now())
    }

    pub fn ban_at(&self, ip: IpAddr, now: Instant) {
        let mut until = self.until.lock().expect("ban lock");
        until.retain(|_, end| *end > now);
        until.insert(ip, now + BAN_TIME);
    }

    pub fn is_banned_at(&self, ip: IpAddr, now: Instant) -> bool {
        self.until.lock().expect("ban lock").get(&ip).is_some_and(|end| *end > now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn the_bucket_allows_a_burst_then_the_rate() {
        let t0 = Instant::now();
        let mut bucket = TokenBucket::new(NonZeroU32::new(20).unwrap(), t0);
        for _ in 0..20 {
            assert_eq!(bucket.take_at(t0), Ok(()));
        }
        // 20 tokens per second: one token every 50 ms
        assert_eq!(bucket.take_at(t0), Err(ms(50)));
        assert_eq!(bucket.take_at(t0 + ms(20)), Err(ms(30)));
        assert_eq!(bucket.take_at(t0 + ms(50)), Ok(()));
        assert_eq!(bucket.take_at(t0 + ms(50)), Err(ms(50)));
        // a long idle period refills one second of tokens, not more
        let later = t0 + Duration::from_secs(60);
        for _ in 0..20 {
            assert_eq!(bucket.take_at(later), Ok(()));
        }
        assert!(bucket.take_at(later).is_err());
    }

    #[test]
    fn a_ban_ends_after_a_day() {
        let t0 = Instant::now();
        let bans = BanList::default();
        let ip: IpAddr = "192.0.2.7".parse().unwrap();
        let other: IpAddr = "192.0.2.8".parse().unwrap();
        assert!(!bans.is_banned_at(ip, t0));
        bans.ban_at(ip, t0);
        assert!(bans.is_banned_at(ip, t0 + BAN_TIME - ms(1)));
        assert!(!bans.is_banned_at(ip, t0 + BAN_TIME));
        assert!(!bans.is_banned_at(other, t0));
    }
}
