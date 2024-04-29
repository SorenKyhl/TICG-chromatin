import math
import os
import os.path as osp

import numpy as np
import pandas as pd
from pylib.utils.utils import load_import_log


def load_rpkm():
    file = '/home/erschultz/dataset_HCT116_RAD21_KO/GSE106886_Rao-2017-Genes.rpkm.txt'
    data = pd.read_csv(file, sep="\t")

    rad21_ko = data.iloc[:,-4:].to_numpy()
    means = np.mean(rad21_ko, axis = 1)
    genes = list(data['Gene Symbol'])

    gene_rpkm_dict = dict(zip(genes, means))
    return gene_rpkm_dict

def gene_info(x):
    # Extract gene names, gene_type, gene_status and level
    g_name = list(filter(lambda x: 'gene_name' in x,  x.split(";")))[0].split("=")[1]
    g_type = list(filter(lambda x: 'gene_type' in x,  x.split(";")))[0].split("=")[1]
    g_status = list(filter(lambda x: 'gene_status' in x,  x.split(";")))[0].split("=")[1]
    g_leve = int(list(filter(lambda x: 'level' in x,  x.split(";")))[0].split("=")[1])
    return (g_name, g_type, g_status, g_leve)

def load_genes():
    # https://medium.com/intothegenomics/annotate-genes-and-genomic-coordinates-using-python-9259efa6ffc2
    file = '/home/erschultz/gencode.v19.annotation.gff3' # hg19
    gencode = pd.read_table(file, comment="#", sep = "\t",
                            names = ['seqname', 'source', 'feature', 'start' ,
                                    'end', 'score', 'strand', 'frame', 'attribute'])

    # Extract Genes in the gff3 file “feature = gene”
    gencode_genes = gencode[(gencode.feature == "gene")][['seqname', 'start', 'end', 'attribute']].copy().reset_index().drop('index', axis=1)
    # Extract gene_name, gene_type, gene_status, level of each gene
    gencode_genes["gene_name"], gencode_genes["gene_type"], gencode_genes["gene_status"], gencode_genes["gene_level"] = zip(*gencode_genes.attribute.apply(lambda x: gene_info(x)))
    # Remove duplicates — Prioritize verified and manually annotated loci over automatically annotated loci
    gencode_genes = gencode_genes.sort_values(['gene_level', 'seqname'], ascending=True).drop_duplicates('gene_name', keep='first').reset_index().drop('index', axis=1)

    return gencode_genes

def load_gene_loci(import_log, pad=0):
    '''
    Filter gencode_genes to region in import_log.

    Inputs:
        import_log: dictionary with details of experimental contact map
        pad: ignore genes with pad bp of simulation boundary
    '''

    gencode_genes = load_genes()

    # filter to chromosome
    chr = import_log['chrom']
    gencode_genes = gencode_genes[(gencode_genes.seqname == f"chr{chr}")][['gene_name', 'start', 'end']]

    # filter to start - end
    sim_start = import_log['start'] + pad
    sim_end = import_log['end'] - pad
    gencode_genes = gencode_genes[(gencode_genes.end > sim_start)]
    gencode_genes = gencode_genes[(gencode_genes.start < sim_end)]

    gene_loci_dict = {}
    resolution = import_log['resolution']
    for index, row in gencode_genes.iterrows():
        name = row['gene_name']
        gene_start = row['start']
        gene_end = row['end']
        start_bead = math.floor(( gene_start - sim_start) / resolution)
        end_bead = math.floor((gene_end - sim_start) / resolution)

        size_b = gene_end - gene_start
        size_kb = np.round(size_b / 1000, 1)
        # print(f'gene: {name}, loci={gene_start}-{gene_end}b ({size_kb}kb), beads={start_bead}-{end_bead}')

        gene_loci_dict[name] = (start_bead, end_bead)

    return gene_loci_dict


def test_load_gene_loci():
    dir = '/home/erschultz/dataset_HCT116_RAD21_KO/samples/sample1'
    result = load_import_log(dir)
    gene_loci_dict = load_gene_loci(result)
    gene_rpkm_dict = load_rpkm()

    found = 0
    total = len(gene_loci_dict)
    for gene in gene_loci_dict.keys():
        if gene in gene_rpkm_dict.keys():
            found += 1
    print(f'found {found} out of {total}')

if __name__ == '__main__':
    # load_rpkm()
    # load_genes()
    test_load_gene_loci()
