import argparse

from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
def main(input_path,output_path):
    tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')
    model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')
    model.eval()
    sentences = []
    with open(input_path, "r", encoding="utf-8") as file:
        for sentence in file:
            sentences.append(sentence.strip())

    masked_sentences = sentences.copy()
    for i in range(len(masked_sentences)):
        masked_sentences[i] = masked_sentences[i].replace("[*]", "[MASK]")

    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    missing_tokens = []
    filled_sentences = []

    for sentence in masked_sentences:
        predictions = fill_mask(sentence)
        filled_sentence = sentence
        tokens=[]
        if isinstance(predictions[0], list):
            for mask_predictions in predictions:
                top_prediction_token = mask_predictions[0]['token_str']
                filled_sentence = filled_sentence.replace("[MASK]", top_prediction_token, 1)
                tokens.append(top_prediction_token)
        else:
            top_prediction_token = predictions[0]['token_str']
            filled_sentence = filled_sentence.replace("[MASK]", top_prediction_token, 1)
            tokens.append(top_prediction_token)
        missing_tokens.append(tokens)
        filled_sentences.append(filled_sentence)

    with open(output_path+"\\dictabert_results.txt", "w",encoding="utf-8") as file:
        for i in range(len(sentences)):
            file.write("Original sentence: " + sentences[i] + "\n")
            file.write("DictaBERT sentence: " + filled_sentences[i] + "\n")
            file.write("DictaBERT tokens: " + str(missing_tokens[i]) + "\n")
            file.write("\n")


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()
    main(args.input_path,args.output_path)
if __name__ == '__main__':
    arg()
