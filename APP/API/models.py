from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


def generateAnxietyResponse(input_text):
    model_path = r"models\anxiety\checkpoint-3500"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Definir la configuración del generador de texto
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    # Definir la semilla para la generación de texto
    prompt = input_text

    # Generar texto a partir de la semilla
    text = generator(prompt, max_length=150, do_sample=True, temperature=0.7)
    print(text[0]['generated_text'])

    return(text[0]['generated_text'])

def generateAnyModelResponse(model_name, input_text):
    model_path = r"models/" + model_name + r"/" + "checkpoint-3500"

    # Alternativa por si hay otros nbombres de checkpoint:
    '''
        folder_path = "models/" + input_text
        model_files = glob.glob(folder_path + "/checkpoint-*")
    '''

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Definir la configuración del generador de texto
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    # Definir la semilla para la generación de texto
    prompt = input_text

    # Generar texto a partir de la semilla
    max_length = 150
    text = generator(prompt, max_length=max_length, do_sample=True, temperature=0.7)

    # Limpieza de texto: eliminar prompt del output y finalizar en un "."
    new_text = text[0]['generated_text'].replace(prompt, "")    # Eliminar el prompt que por defecto se imprime a linicio.
    
    partitioned_string = new_text.rpartition('.')
    before_last_period = partitioned_string[0]+"."

    print(text[0]['generated_text'])

    return(before_last_period)

def generateClassificationPrediction(input_text):

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
   # text_1= 'Ive never taken it to a doctor, but Ive heard that it could help for this problem, but I dont want to take it to the point where I am constantly afraid of dying.'
    pred_1, _ = inference(input_text)

    print("Prediction: {}".format(pred_1[0]))

    return pred_1[0]