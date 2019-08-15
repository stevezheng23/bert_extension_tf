import argparse
import json

def add_arguments(parser):
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)

def eval_sent(input_file,
              output_file):
    with open(input_file, "r") as file:
        input_data = json.load(file)
    
    with open(output_file, "w") as file:
        intent_correct, intent_total = 0, 0
        topic_correct, topic_total = 0, 0
        ability_correct, ability_total = 0, 0
        
        for data in input_data:
            intent_label = data["intent_label"].strip().lower()
            intent_predict = data["intent_predict"].strip().lower()
            topic_label = data["topic_label"].strip().lower()
            topic_predict = data["topic_predict"].strip().lower()
            ability_label = data["ability_label"].strip().lower()
            ability_predict = data["ability_predict"].strip().lower()
            
            if intent_label == intent_predict:
                intent_correct += 1
            if topic_label == topic_predict:
                topic_correct += 1
            if ability_label == ability_predict:
                ability_correct += 1
            
            intent_total += 1
            topic_total += 1
            ability_total += 1
        
        intent_accuracy = (float(intent_correct) / float(intent_total)) if intent_total > 0 else 0.0
        topic_accuracy = (float(topic_correct) / float(topic_total)) if topic_total > 0 else 0.0
        ability_accuracy = (float(ability_correct) / float(ability_total)) if ability_total > 0 else 0.0
        
        file.write("Accuracy (intent-level): {0}\n".format(intent_accuracy))
        file.write("Accuracy (topic-level): {0}\n".format(topic_accuracy))
        file.write("Accuracy (ability-level): {0}\n".format(ability_accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    eval_sent(args.input_file, args.output_file)
