from transformers import BigBirdForTokenClassification, BigBirdTokenizerFast
import torch


config = {'model_name': 'google/bigbird-roberta-base',
         'max_length': 150,
         'train_batch_size':4,
         'valid_batch_size':4,
         'epochs':5,
         'learning_rate':5e-05,
         'max_grad_norm':10,
          'warmup':0.1,
          "grad_acc":8,
          "model_save_path":"big-bird",
         'device': 'cpu'}

output_labels = ['control', 'depression', 'anxiety', 'adhd', 'bpd', 'eda',
       'schizophrenia', 'ptsd', 'bipolar']

tokenizer = BigBirdTokenizerFast.from_pretrained(config['model_name'])
# model = BigBirdForTokenClassification.from_pretrained(config['model_name'],
#                                                      num_labels=len(output_labels))
model = BigBirdForTokenClassification.from_pretrained("models/checkpoint-20250")


labels_to_ids = {v:k for k,v in enumerate(output_labels)}
ids_to_labels = {k:v for k,v in enumerate(output_labels)}

device = config['device']


def inference(sentence):
    inputs = tokenizer(sentence.split(),
                        is_split_into_words=True, 
                        return_offsets_mapping=True, 
                        padding='max_length', 
                        truncation=True, 
                        max_length=200,
                        return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(input_ids=ids, attention_mask=mask, return_dict=False)
#     print(outputs)
    logits = outputs[0]
    
    active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level
    print(logits.shape, active_logits.shape, flattened_predictions.shape)
    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

    prediction = []
    out_str = []
    off_list = inputs["offset_mapping"].squeeze().tolist()
    for idx, mapping in enumerate(off_list):
#         print(mapping, token_pred[1], token_pred[0],"####")

#         only predictions on first word pieces are important
        if mapping[0] == 0 and mapping[1] != 0:
#             print(mapping, token_pred[1], token_pred[0])
            prediction.append(wp_preds[idx][1])
            out_str.append(wp_preds[idx][0])
        else:
            if idx == 1:
                prediction.append(wp_preds[idx][1])
                out_str.append(wp_preds[idx][0])
            continue
    return prediction, out_str


#Probar el modelo:
#Anxiety
text_1= 'Ive never taken it to a doctor, but Ive heard that it could help for this problem, but I dont want to take it to the point where I am constantly afraid of dying.'
pred_1, _ = inference(text_1)

print("Prediction: {}".format(pred_1[0]))