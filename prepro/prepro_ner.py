import argparse
import json
import os.path
import uuid

def add_arguments(parser):
    parser.add_argument("--format", help="format to generate", required=True)
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)

def preprocess(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError("file not found")
    
    processed_data_list = []
    with open(file_name, "r") as file:
        token_list = []
        label_list = []
        for line in file:
            items = [item for item in line.strip().split(' ') if item]
            if len(items) == 0:
                if len(token_list) > 0 and len(label_list) > 0 and len(token_list) == len(label_list):
                    processed_data_list.append({
                        "id": str(uuid.uuid4()),
                        "text": " ".join(token_list),
                        "label": " ".join(label_list)
                    })
                
                token_list.clear()
                label_list.clear()
                continue
            
            if len(items) < 4:
                continue
            
            token = items[0]
            label = items[3]
            
            if token == "-DOCSTART-":
                continue
            
            token_list.append(token)
            label_list.append(label)
    
    return processed_data_list

def output_to_json(data_list, file_name):
    with open(file_name, "w") as file:
        data_json = json.dumps(data_list, indent=4)
        file.write(data_json)

def output_to_plain(data_list, file_name):
    with open(file_name, "wb") as file:
        for data in data_list:
            data_plain = "{0}\t{1}\t{2}\r\n".format(data["id"], data["text"], data["label"])
            file.write(data_plain.encode("utf-8"))

def main(args):
    processed_data = preprocess(args.input_file)
    if (args.format == 'json'):
        output_to_json(processed_data, args.output_file)
    elif (args.format == 'plain'):
        output_to_plain(processed_data, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
