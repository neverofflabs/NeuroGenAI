import pandas as pd
from Bio import SeqIO

# 1. Load SNP CSV (MyHeritage / 23andMe style)
def load_snp_csv(filepath):
    """
    Loads a SNP file with columns: RSID, CHROMOSOME, POSITION, RESULT
    Returns: pandas DataFrame with normalized SNPs
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.upper().str.strip()

    expected_cols = {'RSID', 'CHROMOSOME', 'POSITION', 'RESULT'}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError("Missing required columns in SNP file.")

    # Filter valid nucleotides (A, T, C, G) only
    df = df[df['RESULT'].apply(lambda x: len(str(x)) == 2 and set(str(x)).issubset({'A','T','C','G'}))]
    
    # Normalize fields
    df['CHROMOSOME'] = df['CHROMOSOME'].astype(str).str.replace('"', '').str.strip()
    df = df[df["CHROMOSOME"].isin([str(i) for i in range(1, 23)] + ["X", "Y"])]
    df['POSITION'] = pd.to_numeric(df['POSITION'], errors="coerce").astype("Int64")
    df.dropna(subset=["POSITION"], inplace=True)
    df['POSITION'] = df['POSITION'].astype(int)

    print(f"✅ Loaded SNPs: {df.shape}")
    return df.reset_index(drop=True)

# 2. Load FASTA from NCBI (or other)
def load_fasta(filepath, min_len=20, preview_len=200):
    """
    Parses a FASTA file and returns a DataFrame with ID, Description, Sequence, Length
    """
    records = []
    for record in SeqIO.parse(filepath, "fasta"):
        if len(record.seq) < min_len:
            continue
        records.append({
            "ID": record.id,
            "Description": record.description,
            "Sequence": str(record.seq),
            "Length": len(record.seq),
            "Preview": str(record.seq[:preview_len]) + ("..." if len(record.seq) > preview_len else "")
        })

    df = pd.DataFrame(records)
    print(f"✅ Loaded FASTA sequences: {df.shape}")
    return df