import networkx as nx
import re
import os

def parse_ll_file(filepath):
    """
    Parses the .ll file to extract types from different IR instruction patterns.
    This version attempts to handle various forms of type declarations and uses within the LLVM IR.
    """
    types = []
    type_pattern = re.compile(r"%.* = \w+ ([\[\]\w\s\.\*]+),.*")  # Adjusted to capture more general type patterns
    type_pattern = re.compile(r"%\w+\s+=\s+load\s+(\w+),.*")
    type_pattern = re.compile(r"%\w+\s+=\s+(\w+)")

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            print("line",line)
            match = type_pattern.search(line)
            print(match)
            if match:
                # This captures broader type definitions, including pointers and arrays
                type_info = match.group(1).split()[-1]  # Takes the last word which often is the type
                types.append(type_info)
            else:
                types.append('unknown')  # Use 'unknown' for lines that don't match the pattern

    return types

def load_and_augument_graph(gexf_path, types):
    try:
        G = nx.read_gexf(gexf_path)
        node_ids = list(G.nodes())  # Get a list of node IDs to ensure ordering
        for node_id, node_type in zip(node_ids, types):
            G.nodes[node_id]['type'] = node_type
        return G
    except Exception as e:
        print(f"Failed to read GEXF file: {e}")
        return None  # or handle differently

def main():
    gexf_path = 'gae+pool/pre_dataset/triad.c.gexf'
    ll_path = 'gae+pool/triad.c.ll'

    if not os.path.exists(gexf_path):
        print(f"File not found: {gexf_path}")
        return
    if not os.path.exists(ll_path):
        print(f"File not found: {ll_path}")
        return

    # Parse the .ll file to get only integer types
    types = parse_ll_file(ll_path)

    # Load the graph and augment it with these types
    augmented_graph = load_and_augument_graph(gexf_path, types)

    # Optionally, save the augmented graph back to a file or inspect it
    if augmented_graph:
        nx.write_gexf(augmented_graph, 'gae+pool/dataset/triad.augmented.c.gexf')
        print("Graph augmented and saved successfully.")

    print("Extracted Types:", types)


if __name__ == '__main__':
    main()



