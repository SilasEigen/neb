use super::*;

fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    server.set_read_timeout(Some(Duration::from_millis(200))).unwrap();
    (client, server)
}

// A timeout before the first byte is idle; a timeout inside a header or payload is a framing error.
#[test]
fn a_timeout_inside_a_message_is_not_idle() {
    let magic = [1, 2, 3, 4];
    let (_quiet, mut server) = pair();
    let err = read_message(&mut server, magic).err().expect("timeout");
    assert!(matches!(err.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut));

    let frame = encode_message(magic, CMD_PING, &[0u8; 8]).unwrap();
    for cut in [10, HEADER_BYTES + 3] {
        let (mut client, mut server) = pair();
        client.write_all(&frame[..cut]).unwrap();
        let err = read_message(&mut server, magic).err().expect("timeout");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData, "cut at {cut}");
    }

    let (mut client, mut server) = pair();
    client.write_all(&frame).unwrap();
    assert_eq!(read_message(&mut server, magic).unwrap().unwrap().command, CMD_PING);
}

// A write that times out shuts the socket down: the next write fails at once instead of waiting again.
#[test]
fn a_write_that_times_out_ends_the_session() {
    let (client, server) = pair();
    client.set_write_timeout(Some(Duration::from_millis(200))).unwrap();
    let writer = Arc::new(Mutex::new(client));
    // the server never reads: the socket buffers fill up and a write times out
    let chunk = vec![0u8; 1 << 20];
    let fills = (0..1024).position(|_| write_all(&writer, &chunk).is_err());
    assert!(fills.is_some(), "1 GiB written to a peer that never reads");
    let t = Instant::now();
    assert!(write_all(&writer, &[0u8; 16]).is_err());
    assert!(t.elapsed() < Duration::from_millis(150), "the second write waited {:?}", t.elapsed());
    drop(server);
}

// Inbound sessions leave room for outbound ones and are capped per address; loopback is not capped.
#[test]
fn inbound_peers_leave_room_for_outbound_and_are_capped_per_address() {
    let writer = || Arc::new(Mutex::new(pair().0));
    let at = |a: &str| a.parse::<SocketAddr>().unwrap();
    let set = PeerSet::new(4);
    assert!(set.add(at("10.0.0.1:1"), true, writer()).is_some());
    assert!(set.add(at("10.0.0.1:2"), true, writer()).is_some());
    assert!(set.add(at("10.0.0.1:3"), true, writer()).is_none(), "a third session from one address");
    assert!(set.add(at("10.0.0.2:1"), true, writer()).is_none(), "the last two slots are for outbound sessions");
    assert!(set.add(at("10.0.0.3:1"), false, writer()).is_some());
    assert!(set.add(at("10.0.0.4:1"), false, writer()).is_some());
    assert!(set.add(at("10.0.0.5:1"), false, writer()).is_none(), "all slots taken");
    let set = PeerSet::new(8);
    for port in 1..=6 {
        assert!(set.add(at(&format!("127.0.0.1:{port}")), true, writer()).is_some(), "loopback session {port}");
    }
}
