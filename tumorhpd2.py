##############################################################################
#tumorhpd2 is developed for predicting homing and non homing      #
#protein from their primary sequence. It is developed by Prof G. P. S.       #
#Raghava's group. Please cite : tumorhpd2                                 #
# ############################################################################
import argparse  
import warnings
import pickle
import os
import re
import sys
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, EsmForSequenceClassification, EsmModel
import torch
from torch.utils.data import DataLoader, Dataset
import joblib
import subprocess
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Please provide following arguments. Please make the suitable changes in the envfile provided in the folder.') 

## Read Arguments from command
parser.add_argument("-i", "--input", type=str, required=True, help="Input: protein or peptide sequence in FASTA format or single sequence per line in single letter code")
parser.add_argument("-o", "--output",type=str, default="outfile.csv", help="Output: File for saving results by default outfile.csv")
parser.add_argument("-t","--threshold", type=float, default=0.38, help="Threshold: Value between 0 to 1 by default 0.38")
parser.add_argument("-j","--job", type=int, default=1,choices = [1, 2,3,4],help="Job: 1:Prediction , 2: Design, 3: protein Scan, 4: Motif Search")
parser.add_argument("-x","--dataset", type=int, default=1,choices = [1, 2],help="Job: 1: Whole dataset, 2: length(5-10)")
parser.add_argument("-m","--Model",type=int, default=1, choices = [1, 2], help="Model: 1: RF for job1 or ET for job2, 2: Hybrid, by default 1")
parser.add_argument("-w","--winleng", type=int, choices =range(5, 21), default=7, help="Window Length: 8 to 20 (scan mode only), by default 7")
parser.add_argument("-wd", "--working", type=str, default=os.getcwd(), help="Working directory for intermediate files (optional).")
parser.add_argument("-d","--display", type=int, choices = [1,2], default=2, help="Display: 1:homing, 2: All peptides, by default 2")
args = parser.parse_args()

nf_path = os.path.dirname(os.path.abspath(__file__))

def onehot(ltr):
    return [1 if i == ord(ltr) else 0 for i in range(5, 123)]

def onehotvec(s):
    return [onehot(c) for c in list(s.lower())]

def get_sequence_length(X):
    return max(len(x) for x in X)

def encode_sequences(X, Model, max_seq_length):
    sequence_encode = []

    for i in range(len(X)):
        x = X[i].lower()
        a = onehotvec(x)
        sequence_encode.append(a)

    # Pad or trim sequences to have consistent length of max_seq_length
    sequence_encode = [np.pad(seq, ((0, max_seq_length - len(seq)), (0, 0)), 'constant') for seq in sequence_encode]

    # Convert sequence_encode to a numpy array and cast to float
    sequence_encode = np.array(sequence_encode).astype(np.float32)

    print(sequence_encode.shape)

    # Make predictions using the loaded Model
    input_name = Model.get_inputs()[0].name
    output_name = Model.get_outputs()[0].name
    predictions = Model.run([output_name], {input_name: sequence_encode})

    return predictions[0]

# Function to check the sequence residue
def readseq(file):
    with open(file) as f:
        records = f.read()
    records = records.split('>')[1:]
    seqid = []
    seq = []
    non_standard_detected = False  # Flag to track non-standard amino acids

    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], ''.join(array[1:]).upper()
        
        # Check for non-standard amino acids
        filtered_sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', sequence)
        if filtered_sequence != sequence:
            non_standard_detected = True
        
        seqid.append('' + name)
        seq.append(filtered_sequence)
    
    if len(seqid) == 0:
        f = open(file, "r")
        data1 = f.readlines()
        for each in data1:
            seq.append(each.replace('\n', ''))
        for i in range(1, len(seq) + 1):
            seqid.append("Seq_" + str(i))
    
    if non_standard_detected:
        print("Non-standard amino acids were detected. Processed sequences have been saved and used for further prediction.")
    else:
        print("No non-standard amino acids were detected.")
    
    df1 = pd.DataFrame(seqid)
    df2 = pd.DataFrame(seq)
    return df1, df2

# Function to check the length of sequences and suggest a model
def lenchk(file1):
    cc = []
    df1 = file1
    df1.columns = ['seq']
    
    # Analyze sequence lengths
    for seq in df1['seq']:
        cc.append(len(seq))
    
    # Check if any sequences are shorter than 7
    if any(length < 7 for length in cc):
        raise ValueError("Sequences with length < 7 detected. Please ensure all sequences have length at least 7. Prediction process stopped.")
       
    return df1

# Function for generating pattern of a given length (protein scanning)
def seq_pattern(file1, file2, num):
    df1 = pd.DataFrame(file1, columns=['Seq'])
    df2 = pd.DataFrame(file2, columns=['Name'])

    # Check input lengths
    if df1.empty or df2.empty:
        print("[ERROR] One of the input lists is empty.")
        return pd.DataFrame()

    if len(df1) != len(df2):
        print("[ERROR] Mismatched number of sequences and sequence IDs.")
        print(f"Sequences: {len(df1)}, IDs: {len(df2)}")
        return pd.DataFrame()

    cc, dd, ee, ff, gg = [], [], [], [], []

    for i in range(len(df1)):
        sequence = df1['Seq'][i]
        if not isinstance(sequence, str):
            print(f"[WARNING] Sequence at index {i} is not a string: {sequence}")
            continue

        for j in range(len(sequence)):
            xx = sequence[j:j+num]
            if len(xx) == num:
                cc.append(df2['Name'][i])
                dd.append('Pattern_' + str(j + 1))
                ee.append(xx)
                ff.append(j + 1)  # Start position (1-based)
                gg.append(j + num)  # End position (1-based)

    if not cc:  # Check if any patterns were generated
        print(f"[WARNING] No patterns generated. Possibly all sequences are shorter than {num} residues.")
        return pd.DataFrame()

    df3 = pd.DataFrame({
        'SeqID': cc,
        'Pattern ID': dd,
        'Start': ff,
        'End': gg,
        'Seq': ee
    })

    return df3


def generate_mutant(original_seq, residues, position):
    std = "ACDEFGHIKLMNPQRSTVWY"
    if all(residue.upper() in std for residue in residues):
        if len(residues) == 1:
            mutated_seq = original_seq[:position-1] + residues.upper() + original_seq[position:]
        elif len(residues) == 2:
            mutated_seq = original_seq[:position-1] + residues[0].upper() + residues[1].upper() + original_seq[position+1:]
        else:
            print("Invalid residues. Please enter one or two of the 20 essential amino acids.")
            return None
    else:
        print("Invalid residues. Please enter one or two of the 20 essential amino acids.")
        return None
    return mutated_seq


def generate_mutants_from_dataframe(df, residues, position):
    mutants = []
    for index, row in df.iterrows():
        original_seq = row['seq']
        mutant_seq = generate_mutant(original_seq, residues, position)
        if mutant_seq:
            mutants.append((original_seq, mutant_seq, position))
    return mutants

# Function for generating all possible mutants
def all_mutants(file1,file2):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    cc = []
    dd = []
    ee = []
    df2 = pd.DataFrame(file2)
    df2.columns = ['Name']
    df1 = pd.DataFrame(file1)
    df1.columns = ['Seq']
    for k in range(len(df1)):
        cc.append(df1['Seq'][k])
        dd.append('Original_'+'Seq'+str(k+1))
        ee.append(df2['Name'][k])
        for i in range(0,len(df1['Seq'][k])):
            for j in std:
                if df1['Seq'][k][i]!=j:
                    #dd.append('Mutant_'+df1['Seq'][k][i]+str(i+1)+j+'_Seq'+str(k+1))
                    dd.append('Mutant_'+df1['Seq'][k][i]+str(i+1)+j)
                    cc.append(df1['Seq'][k][:i] + j + df1['Seq'][k][i + 1:])
                    ee.append(df2['Name'][k])
    xx = pd.concat([pd.DataFrame(ee),pd.DataFrame(dd),pd.DataFrame(cc)],axis=1)
    xx.columns = ['SeqID','Mutant_ID','Seq']
    return xx

# ESM2
# Define a function to process sequences

def process_sequences(df, df_2):
    df = pd.DataFrame(df, columns=['seq'])  # Assuming 'seq' is the column name
    df_2 = pd.DataFrame(df_2, columns=['SeqID'])
    # Process the sequences
    outputs = [(df_2.loc[index, 'SeqID'], row['seq']) for index, row in df.iterrows()]
    return outputs


# Function to prepare dataset for prediction
def prepare_dataset(sequences, tokenizer):
    seqs = [seq for _, seq in sequences]
    inputs = tokenizer(seqs, padding=True, truncation=True, return_tensors="pt")
    return inputs


# Function to write output to a file
def write_output(output_file, sequences, predictions, Threshold):
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred:.4f}\n")


# Function to make predictions
def make_predictions(model, inputs, device):
    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs


# Main function for ESM model integration
def run_esm_model(dfseq , df_2, output_file, Threshold):
    # Process sequences from the DataFrame
    sequences = process_sequences(dfseq, df_2)

    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare inputs for the model
    inputs = prepare_dataset(sequences, tokenizer)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Make predictions
    predictions = make_predictions(model, inputs, device)

    # Write the output to a file
    write_output(output_file, sequences, predictions, Threshold)

def filter_and_scale_data(feature_file, selected_cols_file, scaler_path, output_file):
    import pandas as pd
    import joblib

    # Step 1: Read features
    feature_data = pd.read_csv(feature_file, header=0)

    # Step 2: Load selected column names
    with open(selected_cols_file, 'r') as f:
        selected_columns = [line.strip() for line in f if line.strip()]

    # Step 3: Ensure all selected columns exist
    missing_cols = [col for col in selected_columns if col not in feature_data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns from {feature_file}: {missing_cols}")

    # Step 4: Filter features
    filtered_data = feature_data[selected_columns]

    # Step 5: Load scaler or (scaler, columns) tuple
    scaler_obj = joblib.load(scaler_path)
    if isinstance(scaler_obj, tuple):
        scaler, cols_used = scaler_obj
        # Reorder to match training order
        filtered_data = filtered_data[cols_used]
    else:
        scaler = scaler_obj

    # Step 6: Transform data
    scaled_data = scaler.transform(filtered_data.values)

    # Step 7: Save scaled features without header/index
    pd.DataFrame(scaled_data).to_csv(output_file, index=False, header=False)


def prediction(inputfile1, Model, out):
    file_name = inputfile1
    file_name1 = out
    # Load the Model
    clf = joblib.load(Model)
   
    # Load the input data (seq.scaled or similar)
    data_test = np.loadtxt(file_name, delimiter=',')
   
    # Ensure X_test is always 2D, even for a single sequence
    X_test = data_test if len(data_test.shape) > 1 else data_test.reshape(1, -1)
   
    # Get prediction probabilities
    y_p_score1 = clf.predict_proba(X_test)
   
    # Extract the class 1 probabilities
    y_p_s1 = y_p_score1[:, 1]  # Class 1 probabilities (second column)
   
    # Convert to DataFrame and save the probabilities
    df = pd.DataFrame(y_p_s1)
    df.to_csv(file_name1, index=None, header=False)

def class_assignment(file1,thr,out):
    df1 = pd.read_csv(file1, header=None)
    df1.columns = ['ML Score']
    cc = []
    for i in range(0,len(df1)):
        if df1['ML Score'][i]>=float(thr):
            cc.append('homing')
        else:
            cc.append('non-homing')
    df1['Prediction'] = cc
    df1 =  df1.round(3)
    df1.to_csv(out, index=None)

def MERCI_Processor_p(merci_file,merci_processed,name):
    hh =[]
    jj = []
    kk = []
    qq = []
    filename = merci_file
    df = pd.DataFrame(name)
    zz = list(df[0])
    check = '>'
    with open(filename) as f:
        l = []
        for line in f:
            if not len(line.strip()) == 0 :
                l.append(line)
            if 'COVERAGE' in line:
                for item in l:
                    if item.lower().startswith(check.lower()):
                        hh.append(item)
                l = []
    if hh == []:
        ff = [w.replace('>', '') for w in zz]
        for a in ff:
            jj.append(a)
            qq.append(np.array(['0']))
            kk.append('non-homing')
    else:
        ff = [w.replace('\n', '') for w in hh]
        ee = [w.replace('>', '') for w in ff]
        rr = [w.replace('>', '') for w in zz]
        ff = ee + rr
        oo = np.unique(ff)
        df1 = pd.DataFrame(list(map(lambda x:x.strip(),l))[1:])
        df1.columns = ['Name']
        df1['Name'] = df1['Name'].str.strip('(')
        df1[['Seq','Hits']] = df1.Name.str.split("(",expand=True)
        df2 = df1[['Seq','Hits']]
        df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
        df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
        df2.replace(to_replace=r' $', value='', regex=True,inplace=True)
        total_hit = int(df2.loc[len(df2)-1]['Seq'].split()[0])
        for j in oo:
            if j in df2.Seq.values:
                jj.append(j)
                qq.append(df2.loc[df2.Seq == j]['Hits'].values)
                kk.append('homing')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('non-homing')
    df3 = pd.concat([pd.DataFrame(jj),pd.DataFrame(qq),pd.DataFrame(kk)], axis=1)
    df3.columns = ['Name','Hits','Prediction']
    df3.to_csv(merci_processed,index=None)

def Merci_after_processing_p(merci_processed,final_merci_p):
    df5 = pd.read_csv(merci_processed)
    df5 = df5[['Name','Hits']]
    df5.columns = ['Subject','Hits']
    kk = []
    for i in range(0,len(df5)):
        if df5['Hits'][i] > 0:
            kk.append(0.5)
        else:
            kk.append(0)
    df5["MERCI Score (+ve)"] = kk
    df5 = df5[['Subject','MERCI Score (+ve)']]
    df5.to_csv(final_merci_p, index=None)

def MERCI_Processor_n(merci_file,merci_processed,name):
    hh =[]
    jj = []
    kk = []
    qq = []
    filename = merci_file
    df = pd.DataFrame(name)
    zz = list(df[0])
    check = '>'
    with open(filename) as f:
        l = []
        for line in f:
            if not len(line.strip()) == 0 :
                l.append(line)
            if 'COVERAGE' in line:
                for item in l:
                    if item.lower().startswith(check.lower()):
                        hh.append(item)
                l = []
    if hh == []:
        ff = [w.replace('>', '') for w in zz]
        for a in ff:
            jj.append(a)
            qq.append(np.array(['0']))
            kk.append('non-homing')
    else:
        ff = [w.replace('\n', '') for w in hh]
        ee = [w.replace('>', '') for w in ff]
        rr = [w.replace('>', '') for w in zz]
        ff = ee + rr
        oo = np.unique(ff)
        df1 = pd.DataFrame(list(map(lambda x:x.strip(),l))[1:])
        df1.columns = ['Name']
        df1['Name'] = df1['Name'].str.strip('(')
        df1[['Seq','Hits']] = df1.Name.str.split("(",expand=True)
        df2 = df1[['Seq','Hits']]
        df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
        df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
        df2.replace(to_replace=r' $', value='', regex=True,inplace=True)
        total_hit = int(df2.loc[len(df2)-1]['Seq'].split()[0])
        for j in oo:
            if j in df2.Seq.values:
                jj.append(j)
                qq.append(df2.loc[df2.Seq == j]['Hits'].values)
                kk.append('non-homing')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('homing')
    df3 = pd.concat([pd.DataFrame(jj),pd.DataFrame(qq),pd.DataFrame(kk)], axis=1)
    df3.columns = ['Name','Hits','Prediction']
    df3.to_csv(merci_processed,index=None)

def Merci_after_processing_n(merci_processed,final_merci_n):
    df5 = pd.read_csv(merci_processed)
    df5 = df5[['Name','Hits']]
    df5.columns = ['Subject','Hits']
    kk = []
    for i in range(0,len(df5)):
        if df5['Hits'][i] > 0:
            kk.append(-0.5)
        else:
            kk.append(0)
    df5["MERCI Score (-ve)"] = kk
    df5 = df5[['Subject','MERCI Score (-ve)']]
    df5.to_csv(final_merci_n, index=None)


def hybrid(ML_output, name1, seq, merci_output_p, merci_output_n, threshold, final_output):
    # name1 → list of Mutant IDs
    df6_1 = pd.DataFrame(name1, columns=['MutantID'])
    df6_2 = pd.DataFrame(seq, columns=['Sequence'])
    df6_3 = pd.read_csv(ML_output, header=None, names=['ML Score'])

    # MERCI outputs
    df5 = pd.read_csv(merci_output_p, dtype={'Subject': object, 'MERCI Score (+ve)': np.float64})
    df4 = pd.read_csv(merci_output_n, dtype={'Subject': object, 'MERCI Score (-ve)': np.float64})

    # Combine mutant info with ML scores
    df6 = pd.concat([df6_1, df6_2, df6_3], axis=1)
    df6['Subject'] = df6['MutantID'].str.replace('>', '', regex=False)

    # Merge with MERCI scores
    # First merge using MutantID
    df7 = pd.merge(df6, df5.rename(columns={'Subject': 'MutantID'}), 
                on='MutantID', how='outer')

    # Second merge using MutantID too
    df8 = pd.merge(df7, df4.rename(columns={'Subject': 'MutantID'}), 
                on='MutantID', how='outer')
    df8.fillna(0, inplace=True)

    # Calculate hybrid score
    cols = ['ML Score', 'MERCI Score (+ve)', 'MERCI Score (-ve)']
    df8[cols] = df8[cols].apply(pd.to_numeric, errors='coerce')
    df8['Hybrid Score'] = df8[cols].sum(axis=1)

    # Assign predictions
    df8['Prediction'] = ['homing' if score > float(threshold) else 'non-homing'
                         for score in df8['Hybrid Score']]

    # Final clean output
    df8 = df8[['MutantID', 'Sequence', 'ML Score',
               'MERCI Score (+ve)', 'MERCI Score (-ve)',
               'Hybrid Score', 'Prediction']].round(4)

    df8.to_csv(final_output, index=False)


print('##############################################################################')
print('# The program tumorhpd2 is developed for predicting homing and non-homing peptide #')
print("# peptides from their primary sequence, developed by Prof G. P. S. Raghava's group. #")
print('# ############################################################################')

# Parameter initialization or assigning variable for command level arguments

Sequence= args.input        # Input variable 
 
# Output file 
result_filename = args.output
         
# Threshold 
Threshold= float(args.threshold)

# Job
job = args.job

# dataset
dataset = args.dataset

# Model
Model = int(args.Model)

# Window Length 
if args.winleng == None:
    Win_len = int(7)
else:
    Win_len = int(args.winleng)

# Display
dplay = int(args.display)

# Working Directory
wd = args.working


print('Summary of Parameters:')
print('Input File: ',Sequence,'; Model: ',Model,'; Threshold: ', Threshold)
print('Output File: ',result_filename,'; Display: ',dplay)

#------------------ Read input file ---------------------
f=open(Sequence,"r")
len1 = f.read().count('>')
f.close()

with open(Sequence) as f:
        records = f.read()
records = records.split('>')[1:]
seqid = []
seq = []
for fasta in records:
    array = fasta.split('\n')
    name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
    seqid.append(name)
    seq.append(sequence)
if len(seqid) == 0:
    f=open(Sequence,"r")
    data1 = f.readlines()
    for each in data1:
        seq.append(each.replace('\n',''))
    for i in range (1,len(seq)+1):
        seqid.append("Seq_"+str(i))

seqid_1 = list(map(">{}".format, seqid))
CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
CM.to_csv("Sequence_1",header=False,index=None,sep="\n")
f.close()

if job ==1:
    #------------------ Prediction Modeule ---------------------#
    if dataset == 1:
        if Model == 1:
            # 1️⃣ Generate ALLCOMP features using correct path to pfeature_comp.py

            # Run ALLCOMP feature extraction
            subprocess.run(["pfeature_comp", "-i", "Sequence_1", "-j", "ALLCOMP", "-o", "seq.allcomp"], check=True)
            # 2️⃣ Filter & scale features
            filter_and_scale_data(
                feature_file='seq.allcomp',
                selected_cols_file='Model/RF100_data1_feat.txt',
                scaler_path='Model/data1_scaler.pkl',
                output_file='seq.scaled'
            )

            # 3️⃣ Prediction
            prediction('seq.scaled', 'Model/data1_RF100pval_model.pkl', 'seq.pred')

            # 4️⃣ Class assignment
            class_assignment('seq.pred', Threshold, 'seq.out')

            # 5️⃣ Merge results with sequences
            df1 = pd.DataFrame(seqid)
            df2 = pd.DataFrame(seq)
            df3 = pd.read_csv("seq.out").round(3)
            df4 = pd.concat([df1, df2, df3], axis=1)
            df4.columns = ['Subject', 'Sequence', 'ML Score', 'Prediction']

            # Score post-processing
            df4.loc[df4['ML Score'] > 1, 'ML Score'] = 1
            df4.loc[df4['ML Score'] < 0, 'ML Score'] = 0
            df4['PPV'] = (df4['ML Score'] * 1.2341) - 0.1182
            df4.loc[df4['PPV'] > 1, 'PPV'] = 1
            df4.loc[df4['PPV'] < 0, 'PPV'] = 0
            df4 = df4.round({'PPV': 3})

            if dplay == 1:
                df4 = df4.loc[df4.Prediction == "homing"]

            df4.to_csv(result_filename, index=None)

            # 6️⃣ Cleanup
            os.remove('seq.allcomp')
            os.remove('seq.pred')
            os.remove('seq.out')
            os.remove('seq.scaled')
            os.remove('Sequence_1')
        elif Model == 2:
            # Paths to MERCI tools and motifs
            merci = nf_path + '/merci/MERCI_motif_locator.pl'
            motifs_p = nf_path + '/Motifs1/pos_motif.txt'
            motifs_n = nf_path + '/Motifs1/neg_motif.txt'
            # Run ALLCOMP feature extraction
            subprocess.run(["pfeature_comp", "-i", "Sequence_1", "-j", "ALLCOMP", "-o", "seq.allcomp"], check=True)

            # 2️⃣ Filter & scale features
            filter_and_scale_data(
                feature_file='seq.allcomp',
                selected_cols_file='Model/RF100_data1_feat.txt',
                scaler_path='Model/data1_scaler.pkl',
                output_file='seq.scaled'
            )

            # 3️⃣ Prediction
            prediction('seq.scaled', 'Model/data1_RF100pval_model.pkl', 'seq.pred')

            # 4️⃣ Run MERCI motif search
            os.system(f"perl {merci} -p Sequence_1 -i {motifs_p} -o merci_p.txt")
            os.system(f"perl {merci} -p Sequence_1 -i {motifs_n} -o merci_n.txt")

            # 5️⃣ Process MERCI output
            MERCI_Processor_p("merci_p.txt", 'merci_output_p.csv', seqid)
            Merci_after_processing_p('merci_output_p.csv', 'merci_hybrid_p.csv')
            MERCI_Processor_n("merci_n.txt", 'merci_output_n.csv', seqid)
            Merci_after_processing_n('merci_output_n.csv', 'merci_hybrid_n.csv')

            # 6️⃣ Hybrid scoring
            hybrid('seq.pred', seqid, seq, 'merci_hybrid_p.csv', 'merci_hybrid_n.csv', Threshold, 'final_output')

            # 7️⃣ Post-processing Hybrid Score
            df44 = pd.read_csv('final_output')
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
            df44['PPV'] = (df44['Hybrid Score'] * 1.307) - 0.1566
            df44.loc[df44['PPV'] > 1, 'PPV'] = 1
            df44.loc[df44['PPV'] < 0, 'PPV'] = 0
            df44 = df44.round({'PPV': 3})

            if dplay == 1:
                df44 = df44.loc[df44.Prediction == "homing"]

            df44.to_csv(result_filename, index=None)

            # 8️⃣ Cleanup
            os.remove('seq.allcomp')
            os.remove('seq.pred')
            os.remove('seq.scaled')
            os.remove('Sequence_1')
            os.remove('merci_hybrid_p.csv')
            os.remove('merci_hybrid_n.csv')
            os.remove('merci_output_p.csv')
            os.remove('merci_output_n.csv')
            os.remove('merci_p.txt')
            os.remove('merci_n.txt')
            os.remove('final_output')

    elif dataset == 2:
        if Model == 1:
            # 1️⃣ Generate ALLCOMP features using correct path to pfeature_comp.py
            subprocess.run(["pfeature_comp", "-i", "Sequence_1", "-j", "AAC", "-o", "seq.aac"], check=True)

            # 2️⃣ Scale the 20-column features directly (no feature selection)
            feature_data = pd.read_csv('seq.aac', header=0)
            scaler, aac_features = joblib.load("./Model/data2_scaler.pkl")
            scaled_data = scaler.transform(feature_data)
            pd.DataFrame(scaled_data).to_csv('seq.scaled', index=False, header=False)

            # 3️⃣ Prediction
            prediction('seq.scaled', 'Model/data2_ETAAC_model.pkl', 'seq.pred')

            # 4️⃣ Class assignment
            class_assignment('seq.pred', Threshold, 'seq.out')

            # 5️⃣ Merge results with sequences
            df1 = pd.DataFrame(seqid)
            df2 = pd.DataFrame(seq)
            df3 = pd.read_csv("seq.out").round(3)
            df4 = pd.concat([df1, df2, df3], axis=1)
            df4.columns = ['Subject', 'Sequence', 'ML Score', 'Prediction']

            # Score post-processing
            df4.loc[df4['ML Score'] > 1, 'ML Score'] = 1
            df4.loc[df4['ML Score'] < 0, 'ML Score'] = 0
            df4['PPV'] = (df4['ML Score'] * 1.2341) - 0.1182
            df4.loc[df4['PPV'] > 1, 'PPV'] = 1
            df4.loc[df4['PPV'] < 0, 'PPV'] = 0
            df4 = df4.round({'PPV': 3})

            if dplay == 1:
                df4 = df4.loc[df4.Prediction == "homing"]

            df4.to_csv(result_filename, index=None)

            # 6️⃣ Cleanup
            os.remove('seq.aac')
            os.remove('seq.pred')
            os.remove('seq.out')
            os.remove('seq.scaled')
            os.remove('Sequence_1')
        elif Model == 2:
            # Paths to MERCI tools and motifs
            model_save_path = nf_path + '/Model/esm_8M_model'
            merci = nf_path + '/merci/MERCI_motif_locator.pl'
            motifs_p = nf_path + '/Motifs1/pos_motif.txt'
            motifs_n = nf_path + '/Motifs1/neg_motif.txt'
        
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()
        
            # Save sequences for MERCI
            pd.DataFrame(seq).to_csv("Sequence_1", index=False, header=False)
        
            # Path for ESM predictions (equivalent to seq.pred in AAC flow)
            out_file2 = "seq_esm.pred"
        
            # Run ESM-T8 model prediction
            run_esm_model(seq, seqid_1, out_file2, Threshold)
        
            # Run MERCI motif search
            os.system(f"perl {merci} -p Sequence_1 -i {motifs_p} -o merci_p.txt")
            os.system(f"perl {merci} -p Sequence_1 -i {motifs_n} -o merci_n.txt")
        
            # Process MERCI output
            MERCI_Processor_p("merci_p.txt", 'merci_output_p.csv', seqid)
            Merci_after_processing_p('merci_output_p.csv', 'merci_hybrid_p.csv')
            MERCI_Processor_n("merci_n.txt", 'merci_output_n.csv', seqid)
            Merci_after_processing_n('merci_output_n.csv', 'merci_hybrid_n.csv')
        
            # Hybrid scoring
            hybrid(out_file2, seqid, seq, 'merci_hybrid_p.csv', 'merci_hybrid_n.csv', Threshold, 'final_output')
        
            # Post-processing
            df44 = pd.read_csv('final_output')
            df44.rename(columns={"ML Score": "ESM Score"}, inplace=True)
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
            df44['PPV'] = (df44['Hybrid Score'] * 1.307) - 0.1566
            df44.loc[df44['PPV'] > 1, 'PPV'] = 1
            df44.loc[df44['PPV'] < 0, 'PPV'] = 0
            df44 = df44.round({'PPV': 3})
        
            if dplay == 1:
                df44 = df44.loc[df44.Prediction == "homing"]
        
            df44.to_csv(result_filename, index=None)
        
            # Cleanup
            os.remove('Sequence_1')
            os.remove('merci_hybrid_p.csv')
            os.remove('merci_hybrid_n.csv')
            os.remove('merci_output_p.csv')
            os.remove('merci_output_n.csv')
            os.remove('merci_p.txt')
            os.remove('merci_n.txt')
            os.remove('final_output')
            os.remove('seq_esm.pred')


        print('\n======= Thanks for using tumorhpd2. Your results are stored in file :',result_filename,' =====\n\n')
        print('Please cite: tumorhpd2\n')
 
elif job==2:
    #------------------ Design Modeule ---------------------#
    if dataset == 1:
        if Model == 1:
            print(f"\n======= You are using the Design Module of tumorhpd. Results will be saved to: {result_filename} =======\n")
            print("==== Generating mutants and predicting activity using ALLCOMP features... Please wait ====")

            # 1️⃣ Generate mutants from input sequence
            muts = all_mutants(seq, seqid)   # same as in IonNTxPred, returns DataFrame
            muts.to_csv('muts.csv', index=False, header=False)

            # Extract mutant sequences and IDs
            mutant_seqs = muts['Seq'].tolist()
            mutant_ids = muts['Mutant_ID'].tolist()

            # 2️⃣ Write mutants to FASTA-like file (Sequence_1)
            with open("Sequence_1", "w") as f:
                for mid, mseq in zip(mutant_ids, mutant_seqs):
                    f.write(f">{mid}\n{mseq}\n")

            # 3️⃣ Generate ALLCOMP features
            subprocess.run(["pfeature_comp", "-i", "Sequence_1", "-j", "ALLCOMP", "-o", "seq.allcomp"], check=True)

            # 4️⃣ Filter & scale features
            filter_and_scale_data(
                feature_file='seq.allcomp',
                selected_cols_file='Model/RF100_data1_feat.txt',
                scaler_path='Model/data1_scaler.pkl',
                output_file='seq.scaled'
            )

            # 5️⃣ Run prediction
            prediction('seq.scaled', 'Model/data1_RF100pval_model.pkl', 'seq.pred')

            # 6️⃣ Class assignment
            class_assignment('seq.pred', Threshold, 'seq.out')

            # 7️⃣ Merge results with mutant IDs and sequences
            df1 = pd.DataFrame(mutant_ids, columns=['MutantID'])
            df2 = pd.DataFrame(mutant_seqs, columns=['Sequence'])
            df3 = pd.read_csv("seq.out").round(3)
            df4 = pd.concat([df1, df2, df3], axis=1)
            df4.columns = ['MutantID', 'Sequence', 'ML Score', 'Prediction']

            # 8️⃣ Post-process scores
            df4.loc[df4['ML Score'] > 1, 'ML Score'] = 1
            df4.loc[df4['ML Score'] < 0, 'ML Score'] = 0
            df4['PPV'] = (df4['ML Score'] * 1.2341) - 0.1182
            df4.loc[df4['PPV'] > 1, 'PPV'] = 1
            df4.loc[df4['PPV'] < 0, 'PPV'] = 0
            df4 = df4.round({'PPV': 3})

            if dplay == 1:
                df4 = df4.loc[df4.Prediction == "homing"]

            # 9️⃣ Save results
            df4.to_csv(result_filename, index=None)

            # 🔟 Cleanup temporary files
            for f in ['seq.allcomp', 'seq.pred', 'seq.out', 'seq.scaled', 'Sequence_1', 'muts.csv']:
                if os.path.exists(f):
                    os.remove(f)
        elif Model == 2:
                print(f'\n======= You are using the Design Module of TumorHPD. =====\n')
                print('==== Predicting using Hybrid model: Generating mutants, extracting features, please wait ...')

                # 1️⃣ Generate mutants
                muts = all_mutants(seq, seqid_1)
                muts.columns = ['SeqID', 'MutantID', 'Sequence']
                muts.to_csv(f'{wd}/muts.csv', index=False)

                # 2️⃣ Prepare FASTA-like Sequence_1 for pfeature_comp
                with open(f"{wd}/Sequence_1", "w") as outfile:
                    for _, row in muts.iterrows():
                        outfile.write(f">{row['MutantID']}\n{row['Sequence']}\n")

                # 3️⃣ Run ALLCOMP feature extraction
                subprocess.run(["pfeature_comp", "-i", f"{wd}/Sequence_1", "-j", "ALLCOMP", "-o", f"{wd}/seq.allcomp"], check=True)

                # 4️⃣ Filter & scale features
                filter_and_scale_data(
                    feature_file=f"{wd}/seq.allcomp",
                    selected_cols_file=f"{nf_path}/Model/RF100_data1_feat.txt",
                    scaler_path=f"{nf_path}/Model/data1_scaler.pkl",
                    output_file=f"{wd}/seq.scaled"
                )

                # 5️⃣ ML prediction
                prediction(f"{wd}/seq.scaled", f"{nf_path}/Model/data1_RF100pval_model.pkl", f"{wd}/seq.pred")

                # 6️⃣ Run MERCI motif search
                merci = nf_path + '/merci/MERCI_motif_locator.pl'
                motifs_p = nf_path + '/Motifs1/pos_motif.txt'
                motifs_n = nf_path + '/Motifs1/neg_motif.txt'
                os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p} -o {wd}/merci_p.txt")
                os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n} -o {wd}/merci_n.txt")

                # 7️⃣ Process MERCI output
                seqid_list = muts['MutantID'].tolist()
                MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", seqid_list)
                Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
                MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", seqid_list)
                Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")

                # 8️⃣ Hybrid scoring
                hybrid(f"{wd}/seq.pred", seqid_list, muts['Sequence'].tolist(),
                    f"{wd}/merci_hybrid_p.csv", f"{wd}/merci_hybrid_n.csv",
                    Threshold, f"{wd}/final_output")

                # 9️⃣ Post-processing Hybrid Score
                df44 = pd.read_csv(f"{wd}/final_output")
                df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
                df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
                df44['PPV'] = (df44['Hybrid Score'] * 1.307) - 0.1566
                df44.loc[df44['PPV'] > 1, 'PPV'] = 1
                df44.loc[df44['PPV'] < 0, 'PPV'] = 0
                df44 = df44.round({'PPV': 3})
                print(muts.columns)
                print(muts.head())


                # Merge original SeqID for clarity
                df44 = pd.merge(df44, muts[['MutantID', 'SeqID']], on='MutantID', how='left')
                df44 = df44[['SeqID', 'MutantID', 'Sequence', 'ML Score',
                            'MERCI Score (+ve)', 'MERCI Score (-ve)', 'Hybrid Score', 'Prediction', 'PPV']]

                if dplay == 1:
                    df44 = df44.loc[df44.Prediction == "homing"]

                df44.to_csv(result_filename, index=False)

                # 🔟 Cleanup
                temp_files = [
                    'seq.allcomp', 'seq.pred', 'seq.scaled', 'Sequence_1',
                    'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                    'merci_output_p.csv', 'merci_output_n.csv',
                    'merci_p.txt', 'merci_n.txt', 'final_output', 'muts.csv'
                ]
                for file in temp_files:
                    path = f"{wd}/{file}"
                    if os.path.exists(path):
                        os.remove(path)

    elif dataset == 2:
        if Model == 1:
            print(f"\n======= You are using the Design Module of tumorhpd.")
            print("==== Generating mutants and predicting activity using ALLCOMP features... Please wait ====")

            # 1️⃣ Generate mutants from input sequence
            muts = all_mutants(seq, seqid)   # same as in IonNTxPred, returns DataFrame
            muts.to_csv('muts.csv', index=False, header=False)

            # Extract mutant sequences and IDs
            mutant_seqs = muts['Seq'].tolist()
            mutant_ids = muts['Mutant_ID'].tolist()

            # 2️⃣ Write mutants to FASTA-like file (Sequence_1)
            with open("Sequence_1", "w") as f:
                for mid, mseq in zip(mutant_ids, mutant_seqs):
                    f.write(f">{mid}\n{mseq}\n")

            subprocess.run(["pfeature_comp", "-i", "Sequence_1", "-j", "AAC", "-o", "seq.aac"], check=True)

            # 2️⃣ Scale the 20-column features directly (no feature selection)
            feature_data = pd.read_csv('seq.aac', header=0)
            scaler, aac_features = joblib.load("./Model/data2_scaler.pkl")
            scaled_data = scaler.transform(feature_data)
            pd.DataFrame(scaled_data).to_csv('seq.scaled', index=False, header=False)

            # 5️⃣ Run prediction
            prediction('seq.scaled', 'Model/data2_ETAAC_model.pkl', 'seq.pred')

            # 6️⃣ Class assignment
            class_assignment('seq.pred', Threshold, 'seq.out')

            # 7️⃣ Merge results with mutant IDs and sequences
            df1 = pd.DataFrame(mutant_ids, columns=['MutantID'])
            df2 = pd.DataFrame(mutant_seqs, columns=['Sequence'])
            df3 = pd.read_csv("seq.out").round(3)
            df4 = pd.concat([df1, df2, df3], axis=1)
            df4.columns = ['MutantID', 'Sequence', 'ML Score', 'Prediction']

            # 8️⃣ Post-process scores
            df4.loc[df4['ML Score'] > 1, 'ML Score'] = 1
            df4.loc[df4['ML Score'] < 0, 'ML Score'] = 0
            df4['PPV'] = (df4['ML Score'] * 1.2341) - 0.1182
            df4.loc[df4['PPV'] > 1, 'PPV'] = 1
            df4.loc[df4['PPV'] < 0, 'PPV'] = 0
            df4 = df4.round({'PPV': 3})

            if dplay == 1:
                df4 = df4.loc[df4.Prediction == "homing"]

            # 9️⃣ Save results
            df4.to_csv(result_filename, index=None)

            # 🔟 Cleanup temporary files
            for f in ['seq.allcomp', 'seq.pred', 'seq.out', 'seq.scaled', 'Sequence_1', 'muts.csv']:
                if os.path.exists(f):
                    os.remove(f)
        elif Model == 2:
            print(f'\n======= You are using the Design Module of TumorHPD. =====\n')
            print('==== Predicting using Hybrid model (ESM-T8 + MERCI): Generating mutants, please wait ...')
        
            # 1️⃣ Generate mutants
            muts = all_mutants(seq, seqid_1)
            muts.columns = ['SeqID', 'MutantID', 'Sequence']
            muts.to_csv(f'{wd}/muts.csv', index=False)
        
            # 2️⃣ Save sequences for MERCI and ESM
            with open(f"{wd}/Sequence_1", "w") as outfile:
                for _, row in muts.iterrows():
                    outfile.write(f">{row['MutantID']}\n{row['Sequence']}\n")
        
            # 3️⃣ Load ESM-T8 model
            model_save_path = nf_path + '/Model/esm_8M_model'
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()
        
            # 4️⃣ Run ESM predictions
            out_file2 = f"{wd}/seq_esm.pred"
            run_esm_model(muts['Sequence'].tolist(), muts['MutantID'].tolist(), out_file2, Threshold)
        
            # 5️⃣ Run MERCI motif search
            merci = nf_path + '/merci/MERCI_motif_locator.pl'
            motifs_p = nf_path + '/Motifs1/pos_motif.txt'
            motifs_n = nf_path + '/Motifs1/neg_motif.txt'
            os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p} -o {wd}/merci_p.txt")
            os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n} -o {wd}/merci_n.txt")
        
            # 6️⃣ Process MERCI output
            seqid_list = muts['MutantID'].tolist()
            MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", seqid_list)
            Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
            MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", seqid_list)
            Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")
        
            # 7️⃣ Hybrid scoring
            hybrid(out_file2, seqid_list, muts['Sequence'].tolist(),
                f"{wd}/merci_hybrid_p.csv", f"{wd}/merci_hybrid_n.csv",
                Threshold, f"{wd}/final_output")
        
            # 8️⃣ Post-processing Hybrid Score
            df44 = pd.read_csv(f"{wd}/final_output")
            df44.rename(columns={"ML Score": "ESM Score"}, inplace=True)
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
            df44['PPV'] = (df44['Hybrid Score'] * 1.307) - 0.1566
            df44.loc[df44['PPV'] > 1, 'PPV'] = 1
            df44.loc[df44['PPV'] < 0, 'PPV'] = 0
            df44 = df44.round({'PPV': 3})
        
            # Merge original SeqID for clarity
            df44 = pd.merge(df44, muts[['MutantID', 'SeqID']], on='MutantID', how='left')
            df44 = df44[['SeqID', 'MutantID', 'Sequence', 'ESM Score',
                        'MERCI Score (+ve)', 'MERCI Score (-ve)', 'Hybrid Score', 'Prediction', 'PPV']]
        
            if dplay == 1:
                df44 = df44.loc[df44.Prediction == "homing"]
        
            df44.to_csv(result_filename, index=False)
        
            # 🔟 Cleanup
            temp_files = [
                'Sequence_1', 'seq_esm.pred',
                'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'final_output', 'muts.csv'
            ]
            for file in temp_files:
                path = f"{wd}/{file}"
                if os.path.exists(path):
                    os.remove(path)


elif job==3:
    #------------------ protein Scan Modeule ---------------------#
    if dataset == 1:
        if Model == 1:
            print(f'\n======= You are using the Protein Scanning Module of TumorHPD. =====\n')

            # 1️⃣ Generate overlapping peptide patterns
            df_patterns = seq_pattern(seq, seqid_1, Win_len)
            df_patterns["SeqID"] = df_patterns["SeqID"].str.lstrip(">")
            pattern_seqs = df_patterns["Seq"].tolist()
            pattern_ids = df_patterns["SeqID"].tolist()

            # Prepare mutant_ids and mutant_seqs for later merging
            mutant_ids = pattern_ids
            mutant_seqs = pattern_seqs

            # 2️⃣ Prepare FASTA-like Sequence_1
            with open(f"{wd}/Sequence_1", "w") as outfile:
                for pid, pseq in zip(pattern_ids, pattern_seqs):
                    outfile.write(f">{pid}\n{pseq}\n")

            # 3️⃣ Run ALLCOMP feature extraction
            subprocess.run(["pfeature_comp", "-i", f"{wd}/Sequence_1", "-j", "ALLCOMP", "-o", f"{wd}/seq.allcomp"], check=True)

            # 4️⃣ Filter & scale features
            filter_and_scale_data(
                feature_file=f"{wd}/seq.allcomp",
                selected_cols_file=f"{nf_path}/Model/RF100_data1_feat.txt",
                scaler_path=f"{nf_path}/Model/data1_scaler.pkl",
                output_file=f"{wd}/seq.scaled"
            )

            # 5️⃣ ML prediction
            prediction(f"{wd}/seq.scaled", f"{nf_path}/Model/data1_RF100pval_model.pkl", f"{wd}/seq.pred")

            # 6️⃣ Class assignment
            class_assignment('seq.pred', Threshold, 'seq.out')

            # 7️⃣ Merge results with mutant IDs and sequences
            df1 = pd.DataFrame(mutant_ids, columns=['MutantID'])
            df2 = pd.DataFrame(mutant_seqs, columns=['Sequence'])
            df3 = pd.read_csv("seq.out").round(3)
            df4 = pd.concat([df1, df2, df3], axis=1)
            df4.columns = ['MutantID', 'Sequence', 'ML Score', 'Prediction']

            # 8️⃣ Post-process scores
            df4.loc[df4['ML Score'] > 1, 'ML Score'] = 1
            df4.loc[df4['ML Score'] < 0, 'ML Score'] = 0
            df4['PPV'] = (df4['ML Score'] * 1.2341) - 0.1182
            df4.loc[df4['PPV'] > 1, 'PPV'] = 1
            df4.loc[df4['PPV'] < 0, 'PPV'] = 0
            df4 = df4.round({'PPV': 3})

            if dplay == 1:
                df4 = df4.loc[df4.Prediction == "homing"]

            # 9️⃣ Save results
            df4.to_csv(result_filename, index=None)

            # 🔟 Cleanup temporary files
            for f in ['seq.allcomp', 'seq.pred', 'seq.out', 'seq.scaled', 'Sequence_1', 'muts.csv']:
                if os.path.exists(f):
                    os.remove(f)
        elif Model==2:
            print(f'\n======= You are using the Protein Scanning Module of TumorHPD. =====\n')

            # 1️⃣ Generate overlapping peptide patterns
            df_patterns = seq_pattern(seq, seqid_1, Win_len)
            df_patterns["SeqID"] = df_patterns["SeqID"].str.lstrip(">")
            pattern_seqs = df_patterns["Seq"].tolist()
            pattern_ids = df_patterns["SeqID"].tolist()

            # 2️⃣ Prepare FASTA-like Sequence_1
            with open(f"{wd}/Sequence_1", "w") as outfile:
                for pid, pseq in zip(pattern_ids, pattern_seqs):
                    outfile.write(f">{pid}\n{pseq}\n")

            # 3️⃣ Run ALLCOMP feature extraction
            subprocess.run(["pfeature_comp", "-i", f"{wd}/Sequence_1", "-j", "ALLCOMP", "-o", f"{wd}/seq.allcomp"], check=True)

            # 4️⃣ Filter & scale features
            filter_and_scale_data(
                feature_file=f"{wd}/seq.allcomp",
                selected_cols_file=f"{nf_path}/Model/RF100_data1_feat.txt",
                scaler_path=f"{nf_path}/Model/data1_scaler.pkl",
                output_file=f"{wd}/seq.scaled"
            )

            # 5️⃣ ML prediction
            prediction(f"{wd}/seq.scaled", f"{nf_path}/Model/data1_RF100pval_model.pkl", f"{wd}/seq.pred")

            # 6️⃣ Run MERCI motif search
            merci = nf_path + '/merci/MERCI_motif_locator.pl'
            motifs_p = nf_path + '/Motifs1/pos_motif.txt'
            motifs_n = nf_path + '/Motifs1/neg_motif.txt'
            os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p} -o {wd}/merci_p.txt")
            os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n} -o {wd}/merci_n.txt")

            # 7️⃣ Process MERCI output
            muts = df_patterns.rename(columns={'SeqID': 'MutantID', 'Seq': 'Sequence'})
            seqid_list = muts['MutantID'].tolist()
            MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", seqid_list)
            Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
            MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", seqid_list)
            Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")

            # 8️⃣ Hybrid scoring
            hybrid(f"{wd}/seq.pred", seqid_list, muts['Sequence'].tolist(),
                f"{wd}/merci_hybrid_p.csv", f"{wd}/merci_hybrid_n.csv",
                Threshold, f"{wd}/final_output")

            # 9️⃣ Post-processing Hybrid Score
            df44 = pd.read_csv(f"{wd}/final_output")
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
            df44['PPV'] = (df44['Hybrid Score'] * 1.307) - 0.1566
            df44.loc[df44['PPV'] > 1, 'PPV'] = 1
            df44.loc[df44['PPV'] < 0, 'PPV'] = 0
            df44 = df44.round({'PPV': 3})

            # Merge original SeqID for clarity
            df44 = pd.merge(df44, muts[['MutantID', 'Pattern ID']], on='MutantID', how='left')
            df44 = df44.rename(columns={'Pattern ID': 'SeqID'})
            df44 = df44[['SeqID', 'MutantID', 'Sequence', 'ML Score',
             'MERCI Score (+ve)', 'MERCI Score (-ve)', 'Hybrid Score', 'Prediction', 'PPV']]

            if dplay == 1:
                df44 = df44.loc[df44.Prediction == "homing"]

            df44.to_csv(result_filename, index=False)

            # 🔟 Cleanup
            temp_files = [
                'seq.allcomp', 'seq.pred', 'seq.scaled', 'Sequence_1',
                'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'final_output', 'muts.csv'
            ]
            for file in temp_files:
                path = f"{wd}/{file}"
                if os.path.exists(path):
                    os.remove(path)
    if dataset == 2:
        if Model == 1:
            print(f'\n======= You are using the Protein Scanning Module of TumorHPD. =====\n')

            # 1️⃣ Generate overlapping peptide patterns
            df_patterns = seq_pattern(seq, seqid_1, Win_len)
            df_patterns["SeqID"] = df_patterns["SeqID"].str.lstrip(">")
            pattern_seqs = df_patterns["Seq"].tolist()
            pattern_ids = df_patterns["SeqID"].tolist()

            # 2️⃣ Prepare FASTA-like Sequence_1
            with open(f"{wd}/Sequence_1", "w") as outfile:
                for pid, pseq in zip(pattern_ids, pattern_seqs):
                    outfile.write(f">{pid}\n{pseq}\n")

            subprocess.run(["pfeature_comp", "-i", "Sequence_1", "-j", "AAC", "-o", "seq.aac"], check=True)

            # 2️⃣ Scale the 20-column features directly (no feature selection)
            feature_data = pd.read_csv('seq.aac', header=0)
            scaler, aac_features = joblib.load("./Model/data2_scaler.pkl")
            scaled_data = scaler.transform(feature_data)
            pd.DataFrame(scaled_data).to_csv('seq.scaled', index=False, header=False)

            # 5️⃣ ML prediction
            prediction(f"{wd}/seq.scaled", f"{nf_path}/Model/data2_ETAAC_model.pkl", f"{wd}/seq.pred")

            # 6️⃣ Class assignment
            class_assignment('seq.pred', Threshold, 'seq.out')

            # 7️⃣ Merge results with mutant IDs and sequences
            df1 = pd.DataFrame(mutant_ids, columns=['MutantID'])
            df2 = pd.DataFrame(mutant_seqs, columns=['Sequence'])
            df3 = pd.read_csv("seq.out").round(3)
            df4 = pd.concat([df1, df2, df3], axis=1)
            df4.columns = ['MutantID', 'Sequence', 'ML Score', 'Prediction']

            # 8️⃣ Post-process scores
            df4.loc[df4['ML Score'] > 1, 'ML Score'] = 1
            df4.loc[df4['ML Score'] < 0, 'ML Score'] = 0
            df4['PPV'] = (df4['ML Score'] * 1.2341) - 0.1182
            df4.loc[df4['PPV'] > 1, 'PPV'] = 1
            df4.loc[df4['PPV'] < 0, 'PPV'] = 0
            df4 = df4.round({'PPV': 3})

            if dplay == 1:
                df4 = df4.loc[df4.Prediction == "homing"]

            # 9️⃣ Save results
            df4.to_csv(result_filename, index=None)

            # 🔟 Cleanup temporary files
            for f in ['seq.allcomp', 'seq.pred', 'seq.out', 'seq.scaled', 'Sequence_1', 'muts.csv']:
                if os.path.exists(f):
                    os.remove(f)
        elif Model == 2:
            print(f'\n======= You are using the Protein Scanning Module of TumorHPD. =====\n')
        
            # 1️⃣ Generate overlapping peptide patterns
            df_patterns = seq_pattern(seq, seqid_1, Win_len)
            df_patterns["SeqID"] = df_patterns["SeqID"].str.lstrip(">")
            pattern_seqs = df_patterns["Seq"].tolist()
            pattern_ids = df_patterns["SeqID"].tolist()
        
            # 2️⃣ Save sequences for MERCI and ESM
            with open(f"{wd}/Sequence_1", "w") as outfile:
                for pid, pseq in zip(pattern_ids, pattern_seqs):
                    outfile.write(f">{pid}\n{pseq}\n")
        
            # 3️⃣ Load ESM-T8 model
            model_save_path = nf_path + '/Model/esm_8M_model'
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()
        
            # 4️⃣ Run ESM predictions
            out_file2 = f"{wd}/seq_esm.pred"
            run_esm_model(pattern_seqs, pattern_ids, out_file2, Threshold)
        
            # 5️⃣ Run MERCI motif search
            merci = nf_path + '/merci/MERCI_motif_locator.pl'
            motifs_p = nf_path + '/Motifs1/pos_motif.txt'
            motifs_n = nf_path + '/Motifs1/neg_motif.txt'
            os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p} -o {wd}/merci_p.txt")
            os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n} -o {wd}/merci_n.txt")
        
            # 6️⃣ Process MERCI output
            seqid_list = pattern_ids
            MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", seqid_list)
            Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
            MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", seqid_list)
            Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")
        
            # 7️⃣ Hybrid scoring
            hybrid(out_file2, seqid_list, pattern_seqs,
                   f"{wd}/merci_hybrid_p.csv", f"{wd}/merci_hybrid_n.csv",
                   Threshold, f"{wd}/final_output")
        
            # 8️⃣ Read hybrid output & rename column
            df44 = pd.read_csv(f"{wd}/final_output")
            df44.rename(columns={"ML Score": "ESM Score"}, inplace=True)
        
            # 🔹 Safely merge SeqID mapping instead of inserting list
            id_map = pd.DataFrame({"Sequence": pattern_seqs, "SeqID": pattern_ids})
            df44 = pd.merge(df44, id_map, on="Sequence", how="left")
        
            # 9️⃣ Post-processing Hybrid Score
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
            df44['PPV'] = (df44['Hybrid Score'] * 1.307) - 0.1566
            df44.loc[df44['PPV'] > 1, 'PPV'] = 1
            df44.loc[df44['PPV'] < 0, 'PPV'] = 0
            df44 = df44.round({'PPV': 3})
        
            # Final column order
            df44 = df44[['SeqID', 'Sequence', 'ESM Score',
                         'MERCI Score (+ve)', 'MERCI Score (-ve)',
                         'Hybrid Score', 'Prediction', 'PPV']]
        
            if dplay == 1:
                df44 = df44.loc[df44.Prediction == "homing"]
        
            df44.to_csv(result_filename, index=False)
        
            # 🔟 Cleanup
            temp_files = [
                'Sequence_1', 'seq_esm.pred',
                'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'final_output'
            ]
            for file in temp_files:
                path = f"{wd}/{file}"
                if os.path.exists(path):
                    os.remove(path)



elif job == 4:
    print(f'\n======= You are using the Motif Scanning module of TumorHPD2.=====\n')
    df_2, dfseq = readseq(Sequence)
    df1 = lenchk(dfseq)

    merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"

    if dataset == 1:
        pos_motif_file = f"{nf_path}/Motifs1/pos_motif.txt"
        neg_motif_file = f"{nf_path}/Motifs1/neg_motif.txt"
    elif dataset == 2:
        pos_motif_file = f"{nf_path}/Motifs2/pos_motif.txt"
        neg_motif_file = f"{nf_path}/Motifs2/neg_motif.txt"
    else:
        raise ValueError(f"Unsupported dataset value: {dataset}")

    os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {pos_motif_file} -o {wd}/merci_p.txt")
    os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {neg_motif_file} -o {wd}/merci_n.txt")

    # Process MERCI results
    MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", df_2)
    Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
    MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", df_2)
    Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")

    # Read each CSV file into a separate DataFrame
    df_p = pd.read_csv(f"{wd}/merci_output_p.csv")
    df_n = pd.read_csv(f"{wd}/merci_output_n.csv")

    # Rename columns for clarity
    df_p = df_p.rename(columns={'Name': 'SeqID', 'Hits': 'PHits', 'Prediction': 'PPrediction'})
    df_n = df_n.rename(columns={'Name': 'SeqID', 'Hits': 'NHits', 'Prediction': 'NPrediction'})

    # Merge the DataFrames on 'SeqID'
    df_merged = df_p.merge(df_n, on='SeqID', how='outer').fillna(0)

    df4_selected = df_merged[['SeqID', 'PHits', 'NHits']].copy()

    # Define prediction function
    def determine_prediction(row):
        if row['PHits'] == 0 and row['NHits'] == 0:
            return 'non-homing'
        elif row['PHits'] > row['NHits']:
            return 'homing'
        elif row['PHits'] < row['NHits']:
            return 'Non-homing'
        elif row['PHits'] == row['NHits']:
            return 'homing'
        else:
            return 'NA'

    # Apply prediction
    df4_selected['Prediction'] = df4_selected.apply(determine_prediction, axis=1)

    # Rename columns
    df4_selected.columns = ["SeqID", 'Positive Hits', 'Negative Hits', "Prediction"]

    # Save results
    df4_selected.to_csv(result_filename, index=False)

    # Display if requested
    if dplay == 1:
        df_filtered = df4_selected.loc[df4_selected.Prediction == "homing"]
        print(df_filtered)
    elif dplay == 2:
        print(df4_selected)

    # Clean up temporary files
    temp_files = [
        'final_output', 'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
        'merci_output_p.csv', 'merci_output_n.csv',
        'merci_p.txt', 'merci_n.txt', 'out22', 'Sequence_1'
    ]
    for file in temp_files:

        path = f"{wd}/{file}"
        if os.path.exists(path):
            os.remove(path)
