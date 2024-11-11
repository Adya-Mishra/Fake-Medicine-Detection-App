# import pandas as pd

# # Load your dataset
# df = pd.read_csv("fraud_platform_imen/data/medicine_data.csv")

# # List of columns to keep
# columns_to_keep = ['product_name', 'sub_category', 'salt_comp', 'manufactured_by']

# # Drop columns that are not in the list
# df = df[columns_to_keep]

# # Remove duplicate rows
# df_cleaned = df.drop_duplicates()

# # Save the cleaned dataset
# df_cleaned.to_csv("cleaned_medicine_data.csv", index=False)



# import pandas as pd
# import spacy
# from spacy.tokens import DocBin
# from tqdm import tqdm

# # Step 1: Load the CSV file
# csv_file_path = "cleaned_medicine_data.csv"  # Adjust the path to your actual CSV file
# df = pd.read_csv(csv_file_path)

# # Step 2: Initialize spaCy's blank language model and DocBin
# nlp = spacy.blank("en")  # Load a blank spaCy model
# doc_bin = DocBin()  # Create a DocBin container for storing Doc objects

# # Step 3: Define a function to create entity spans
# def create_entity_spans(text, entities_info):
#     """
#     text: Concatenated text from product_name, sub_category, etc.
#     entities_info: A list of tuples (start, end, label) representing entities
#     """
#     doc = nlp.make_doc(text)
#     ents = []
    
#     # Sorting entities by their start index to ensure no overlaps
#     entities_info = sorted(entities_info, key=lambda x: x[0])
    
#     for start, end, label in entities_info:
#         # Check for overlap before adding the entity span
#         # We only add non-overlapping spans
#         if not any(ent.start <= start < ent.end or ent.start < end <= ent.end for ent in ents):
#             span = doc.char_span(start, end, label=label)
#             if span is not None:
#                 ents.append(span)
    
#     doc.ents = ents
#     return doc

# # Step 4: Iterate through the CSV and convert each row to a training example
# for i, row in tqdm(df.iterrows(), total=df.shape[0]):
#     # Ensure all fields are converted to strings
#     product_name = str(row['product_name']) if pd.notna(row['product_name']) else ""
#     sub_category = str(row['sub_category']) if pd.notna(row['sub_category']) else ""
#     salt_comp = str(row['salt_comp']) if pd.notna(row['salt_comp']) else ""
#     manufactured_by = str(row['manufactured_by']) if pd.notna(row['manufactured_by']) else ""
    
#     # Concatenate the text fields into one string
#     text = f"{product_name} {sub_category} {salt_comp} {manufactured_by}"
    
#     # Calculate entity spans
#     entities_info = []
    
#     # For product_name
#     start = text.find(product_name)
#     end = start + len(product_name)
#     entities_info.append((start, end, 'product_name'))
    
#     # For sub_category
#     start = text.find(sub_category)
#     end = start + len(sub_category)
#     entities_info.append((start, end, 'sub_category'))
    
#     # For salt_comp
#     start = text.find(salt_comp)
#     end = start + len(salt_comp)
#     entities_info.append((start, end, 'salt_comp'))
    
#     # For manufactured_by
#     start = text.find(manufactured_by)
#     end = start + len(manufactured_by)
#     entities_info.append((start, end, 'manufactured_by'))
    
#     # Create a spaCy Doc object with entity annotations
#     doc = create_entity_spans(text, entities_info)
    
#     # Add the Doc to the DocBin
#     doc_bin.add(doc)

# # Step 5: Save the DocBin to disk
# doc_bin_path = "train.spacy"  # Adjust the output path
# doc_bin.to_disk(doc_bin_path)
# print(f"Saved DocBin to {doc_bin_path}")


# import pandas as pd
# import spacy
# from spacy.tokens import DocBin

# # Load the data (e.g., your cleaned CSV file)
# df = pd.read_csv('cleaned_medicine_data.csv')

# # Initialize a spaCy model (you can use an existing one or create a blank one)
# nlp = spacy.blank("en")

# # Initialize DocBin (a container for spaCy Doc objects)
# doc_bin = DocBin()

# # Split the data into train and dev (validation) sets (80-20 split)
# train_data = df.sample(frac=0.8, random_state=42)
# dev_data = df.drop(train_data.index)

# def create_entity_spans(text, entities_info):
#     # Function to create a spaCy Doc and add entities
#     doc = nlp.make_doc(text)
#     entities = []
    
#     # Create a set to track token indices that are already part of an entity span
#     token_indices = set()

#     for ent_info in entities_info:
#         start = text.find(ent_info['text'])
#         end = start + len(ent_info['text'])
        
#         # Check if the span already exists (to avoid overlap)
#         span = doc.char_span(start, end, label=ent_info['label'])
        
#         # Only add the span if it doesn't overlap with an existing entity
#         if span is not None:
#             # Check for token overlaps
#             if all(token.i not in token_indices for token in span):
#                 entities.append(span)
#                 # Mark tokens in this span as used
#                 for token in span:
#                     token_indices.add(token.i)
    
#     doc.ents = entities  # Assign entities to the doc
#     return doc

# # Create dev.spacy data (using the dev_data subset)
# for _, row in dev_data.iterrows():
#     # Convert all values to strings before concatenating
#     text = str(row['product_name']) + " " + str(row['sub_category']) + " " + str(row['salt_comp']) + " " + str(row['manufactured_by'])
#     entities_info = [
#         {"text": str(row['product_name']), "label": "product_name"},
#         {"text": str(row['sub_category']), "label": "sub_category"},
#         {"text": str(row['salt_comp']), "label": "salt_comp"},
#         {"text": str(row['manufactured_by']), "label": "manufactured_by"}
#     ]
#     doc = create_entity_spans(text, entities_info)
#     doc_bin.add(doc)  # Correct method name is `add()`

# # Save dev.spacy file
# doc_bin.to_disk('dev.spacy')




# import spacy

# # Load the trained model
# nlp = spacy.load('output\model-last')

# # Test the model on new data
# text = "Rx Oridazole and Diloxanide Furoate Tablets amicline PLUSC 3ftari- TT10 X 10 tablets in blister pack FORMULA: Each compression coated tablet contains: Ornidazole I.P_ 250 mg Diloxanide Furoate L.P 375 mg Excipients ~q-S. Colour Quinoline Yellow WS (C.I: No. 47005) (in the coat) Store below 30'C_ Protect from light and moisture. Keep out of reach of children_"
# doc = nlp(text)

# # Print the named entities detected by the model
# for ent in doc.ents:
#     print(ent.text, ent.label_)




# import spacy
# from spacy.training import Example
# from spacy.tokens import DocBin

# # Load the trained model
# nlp = spacy.load('./output/model-last')

# # Load dev data
# test_data_path = 'dev.spacy'  # Adjust the path to your .spacy file
# doc_bin = DocBin().from_disk(test_data_path)

# # Convert to examples for evaluation
# examples = []
# for doc in doc_bin.get_docs(nlp.vocab):
#     example = Example.from_dict(doc, {"entities": doc.ents})
#     examples.append(example)

# # Evaluate the model
# metrics = nlp.evaluate(examples)

# # Print the evaluation metrics
# print(metrics)



import spacy

# Load the trained model (you can choose 'model-best' or 'model-last')
nlp = spacy.load('output/model-last')  # or 'output/model-best'
# Save the model to a new location
nlp.to_disk('my_saved_model')



