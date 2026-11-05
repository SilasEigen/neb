//! The recorded chain fixtures (`pow/tests/chains/pools-mnist.txt`) name the dataset file that the network
//! parameters pin: a replay with another file would be refused by `FileDataset::load`.

use neb_node::params::DATASET_HASH;

#[test]
fn the_recorded_chain_fixtures_name_the_dataset_of_the_network() {
    let want: String = DATASET_HASH.iter().map(|b| format!("{b:02x}")).collect();
    {
        let name = "pools-mnist";
        let path = format!("{}/../../../pow/tests/chains/{name}.txt", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
        let line = text.lines().nth(2).unwrap();
        assert_eq!(line, format!("dataset file {want}"), "{name}");
    }
}
