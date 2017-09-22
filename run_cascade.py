#!/usr/bin/env python

from __future__ import print_function
import click
import os.path as P
import csv
import sys
import numpy as np
import networkx as nx
import random as R

exp_r = np.random.exponential
SEED = 42


@click.command()
@click.option('-o', '--output', 'output_path',
              prompt='Which file to write cascades to',
              default='cascades.csv',
              help='File to write cascade to.')
@click.option('-i', '--input', 'input_file',
              prompt='Which file to read input graph from',
              type=click.Path(exists=True),
              help='Input graph (directed with weights).')
@click.option('-c', '--cascades',
              default=10,
              help='Number of cascades to generate.')
@click.option('-s', '--seed',
              default=SEED,
              help='Seed for the random number generator.')
@click.option('-f', '--force',
              is_flag=True,
              help='Force to overwrite files.')
@click.option('-T', '--time', 'time_period',
              default=1.0,
              help='Time period for which to run.')
def run(output_path, input_file,
        cascades, seed, force, time_period):

    any_op_file_exists = P.exists(output_path)

    if any_op_file_exists and not force:
        print('Cannot overwrite output file without --force', file=sys.stderr)
        sys.exit(-1)

    g = nx.read_edgelist(input_file, create_using=nx.DiGraph())

    # reduce source of randomness here.
    nodes = sorted(g.nodes())
    cascade_data = []
    R.seed(seed)
    for idx in range(cascades):
        cur_time = 0
        src = R.choice(nodes)
        # TODO: Generate different infection models.
        ticking_edges = [(exp_r(1.0 / e[2]['act_prob']), e)
                         for e in g.out_edges(src, data=True)]
        infected_nodes = set([ src ])
        cascade_data.append({
            'cascade_id': idx,
            'dst': src, # Initially, the 'src' is infected.
            'at': 0
        })
        while True:
            # - If the whole network (or reachable component) was infected, then stop
            if len(ticking_edges) == 0:
                break

            # - Calculate the next "tick" and the edge which ticked.
            dt, inf_edge = sorted(ticking_edges)[0]
            cur_time = cur_time + dt

            # - Record data
            if cur_time < time_period:
                cascade_data.append({
                    'cascade_id': idx,
                    'dst': inf_edge[1],
                    'at': cur_time
                })
            else:
                break

            # - Add all out_edges from dst to ticking_edges
            # TODO: Generate different infection models.
            ticking_edges += [(exp_r(1.0 / e[2]['act_prob']), e)
                              for e in g.out_edges(inf_edge[1], data=True)]

            # - Add source to infected_nodes
            infected_nodes.add(inf_edge[1])

            # - Remove edges from ticking_edges which go to infected_nodes
            ticking_edges = [edge for edge in ticking_edges
                             if edge[1][1] not in infected_nodes]

    with open(output_path, 'w') as output_file:
        w = csv.DictWriter(output_file, ['cascade_id', 'dst', 'at'])
        w.writeheader()
        w.writerows(cascade_data)


if __name__ == '__main__':
    run()

