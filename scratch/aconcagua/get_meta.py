import sys, pyarrow.parquet as pq, pyarrow.fs as pfs, pyarrow as pa, pyarrow.compute as pc
fs=pfs.S3FileSystem(anonymous=True,region="us-west-2")
f=pq.ParquetFile(fs.open_input_file("ops-explorer-public/aconcagua/aconcagua/cell_data.parquet"),pre_buffer=True)
cols=["plate","well","tile","cell_phenotype","gene_symbol_1","perturbation_id","mapped_single_gene","class","is_control","region","mapped_n_barcodes","distance","cell_barcode_1","barcode"]
w=None
for i in range(f.num_row_groups):
    t=f.read_row_group(i,columns=cols)
    t=pa.table({c:(pc.dictionary_encode(t[c]) if pa.types.is_string(t[c].type) or pa.types.is_large_string(t[c].type) else t[c]) for c in cols})
    t=t.cast(pa.schema([pa.field(c,pa.dictionary(pa.int32(),pa.string()) if pa.types.is_dictionary(t[c].type) else t[c].type) for c in cols]))
    w=w or pq.ParquetWriter(sys.argv[1],t.schema)
    w.write_table(t)
    if i%20==0: print(i,flush=True)
w.close()
