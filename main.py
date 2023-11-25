from datasets import load_dataset
import numpy as np
from datasets import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import argparse
import torch.nn.functional as F
import json
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
import time
import clip
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score,precision_recall_curve,auc
import os

parser = argparse.ArgumentParser(description = "Train a CLIP model on the pathVQA dataset")

#Arguments
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--train_batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--valid_batch_size', type=int, default=128, help='Batch size for validation and test')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for the optimizer')
parser.add_argument('--seed', type=int, default=23, help='Random seed for reproducibility')
parser.add_argument('--train', type=bool, default=False, help='Whether to train the model')
parser.add_argument('--operation',type=str,default='concat',help='Operation to use for combining image and text features (concat, add)')
parser.add_argument('--modelname',type=str,default='ViT-B32',help='CLIP model to use (ViT-B/32, RN50)')
parser.add_argument('--device',type=str,default='cuda',help='Device to use for training')
parser.add_argument('--weight_decay',type=float,default=0.2,help='Weight decay for the optimizer')
parser.add_argument('--betas',type=tuple,default=(0.9,0.98),help='Betas for the optimizer')
parser.add_argument('--epsilon',type=float,default=1e-6,help='Epsilon for the optimizer')
parser.add_argument('--num_layers',type=int,default=1,help='Number of layers for the classifier')
parser.add_argument('--hidden_dim',type=int,default=1024,help='Hidden dimension for the classifier')
parser.add_argument('--loss_func',type=str,default='BCELoss',help='Loss function to use (BCELoss, MSELoss)')
parser.add_argument('--max_length',type=int,default=40,help='Maximum number of tokens to use for the question')
parser.add_argument('--mode',type=str,default='None',help="Mode to test or demo")
parser.add_argument('--img' ,type=str,default='None',help="Image to test")
parser.add_argument('--bestmethod',type=bool,default=False,help="If testing best method change plot titles")
parser.add_argument('--pretrained',type=bool,default=False,help="If testing with pretrained weights (using the full path in BCV002). False for using the final model in the current directory (requires training first).")
args = parser.parse_args()

#Set experiment name, depending on the arguments, shortening to Final Method if the best method is evaluated for a better plot title in the PR curve
if args.bestmethod == False:
    experiment_name = f"CLIP_{args.modelname}_{args.operation}_nl{args.num_layers}_hd{args.hidden_dim}_{args.loss_func}_w{args.max_length}"
elif args.bestmethod == True:
    experiment_name = "Final Method"

#Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load dataset, directly from huggingface
dataset = load_dataset("flaviagiammarino/path-vqa")

#Function that filters the dataset to only include yes/no questions
def filter_function(example):
    return example["answer"] in ["yes", "no"]

#Function that tokenizes the question depending on the max number of tokens and pads to length 77 for clip text encoder.
def process_question(question):
    tokens = clip.tokenize([question])
    tokens = tokens[:, :args.max_length]
    padding = 77 - tokens.shape[1] #Sequences must be of length 77 for clip text encoder
    if padding > 0:
        tokens = F.pad(tokens, (0, padding), "constant", 0)
    return tokens

#Function that one-hot encodes the answer
def process_answer(answer):
    return 1 if answer == "yes" else 0

#Filtering datasets to include only yes/no questions
filtered_train = dataset["train"].filter(filter_function)
filtered_valid = dataset["validation"].filter(filter_function)
filtered_test = dataset["test"].filter(filter_function)

#Transformations to apply to the images
image_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((224, 224)),  # Resize to (224,224) for CLIP image encoder
    transforms.ToTensor(),
])

#Function to collate the data into batches for dataloaders
def collate_fn(batch):
    images = torch.stack([torch.tensor(item['image']) for item in batch])
    questions = torch.stack([torch.tensor(item['question']) for item in batch])
    answers = torch.stack([torch.tensor(item['answer']) for item in batch])
    return {'image': images, 'question': questions, 'answer': answers}

#Create dataloaders and process datasets depending on whether the model will be trained or only evaluated.
if args.train == True: #If training, process train and valid images
    print("Processing train images...")
    resized_filtered_train = Dataset.from_dict({
        'image': [image_transform(image) for image in tqdm(filtered_train['image'])],
        'question':[process_question(question) for question in tqdm(filtered_train['question'])],
        'answer': [process_answer(answer) for answer in filtered_train['answer']],
    })
    print("Processing valid images...")
    resized_filtered_valid = Dataset.from_dict({
        'image':[image_transform(image) for image in tqdm(filtered_valid['image'])],
        'question':[process_question(question) for question in tqdm(filtered_valid['question'])],
        'answer': [process_answer(answer) for answer in filtered_valid['answer']],
    })
    train_loader = DataLoader(resized_filtered_train,batch_size=args.train_batch_size, collate_fn=collate_fn,shuffle=True)
    valid_loader = DataLoader(resized_filtered_valid,batch_size=args.valid_batch_size, collate_fn=collate_fn)

if args.train == False and args.mode == "valid": #If evaluating on the validation set, process valid images
    print("Processing valid images...")
    resized_filtered_valid = Dataset.from_dict({
        'image':[image_transform(image) for image in tqdm(filtered_valid['image'])],
        'question':[process_question(question) for question in tqdm(filtered_valid['question'])],
        'answer': [process_answer(answer) for answer in filtered_valid['answer']],
    })
    valid_loader = DataLoader(resized_filtered_valid,batch_size=args.valid_batch_size, collate_fn=collate_fn)


class CustomCLIPModel(nn.Module):#CLIP model with a custom classifier
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        layers = []
        if args.operation == 'concat' and args.modelname == 'ViT-B32': #CLIP ViT-B32 outputs 512-dimensional image and text features
            input_dim = 1024
        elif args.operation == 'add' and args.modelname == 'ViT-B32':
            input_dim = 512
        elif args.operation == 'concat' and args.modelname == 'RN50': #CLIP RN50 outputs 1024-dimensional image and text features
            input_dim = 2048
        elif args.operation == 'add' and args.modelname == 'RN50':
            input_dim = 1024
        for _ in range(args.num_layers): #Add the number of layers specified in the arguments with the hidden dimensions specified as well
            layers.append(nn.Linear(input_dim, args.hidden_dim))
            layers.append(nn.ReLU())
            input_dim = args.hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid()) #Output a single value between 0 and 1
        self.fc = nn.Sequential(*layers)

    def forward(self, image, text):
        image_features = self.clip_model.encode_image(image).float()
        text_features = self.clip_model.encode_text(text).float()
        
        if args.operation == 'concat': #Concatenate or add features, depending on arguments
            features = torch.cat((image_features, text_features), dim=1)
        elif args.operation == 'add':
            features = image_features + text_features
        return self.fc(features)

#Load and initialize the model with the requested architecture.
modelonombre = 'ViT-B/32' if args.modelname == 'ViT-B32' else args.modelname
clip_model, preprocess = clip.load(modelonombre, device=device, jit=False)
model = CustomCLIPModel(clip_model).to(device)
if args.modelname == 'RN50':
    clip_model.float()

#Load the requested loss function
if args.loss_func == "BCELoss":
    loss_func = nn.BCELoss()
elif args.loss_func == "MSELoss":
    loss_func = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,betas=args.betas,eps=args.epsilon,weight_decay=args.weight_decay)

# Initialize a dictionary to store the accuracy, precision, and recall results
accuracy_results = {}
#TRAINING LOOP
if args.train == True and args.mode == "None":
    best_val_fscore = float('-inf') #We will save the model with the best Fscore
    num_epochs = 10
    print("Training...")
    num_batches = len(train_loader)
    for epoch in range(num_epochs):
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            images = batch["image"].to(device).half()
            questions = batch["question"].squeeze(1).to(device)
            answers = batch["answer"].unsqueeze(1).to(device).float()
            outputs = model(images, questions)
            loss = loss_func(outputs, answers)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            elapsed_time = time.time() - start_time
            batches_done = i + 1
            batches_left = num_batches - batches_done
            time_per_batch = elapsed_time / batches_done
            estimated_time_left = batches_left * time_per_batch

            minutes, seconds = divmod(estimated_time_left, 60)

            print(f'Epoch: {epoch+1}, Batch {batches_done}/{num_batches}, Loss: {loss.item()}, Estimated time left: {int(minutes)} minutes {int(seconds)} seconds')

            # Evaluate at the end of each Epoch
            if i == len(train_loader) - 1:
                print("Evaluating...")
                correct = 0
                total = 0
                all_predicted = [] #Used to obtain accuracy, precision, and recall
                all_answers = []
                num_eval_batches = len(valid_loader)
                with torch.no_grad():
                    for j, batch in enumerate(valid_loader):
                        images = batch["image"].to(device).half()
                        questions = batch["question"].squeeze(1).to(device)
                        answers = batch["answer"].unsqueeze(1).to(device).float()
                        outputs = model(images, questions)
                        predicted = (outputs > 0.5).float()  # Threshold the outputs

                        print(f'Evaluated batches: {j+1}/{num_eval_batches}, Remaining batches: {num_eval_batches - (j+1)}')

                        total += answers.size(0)
                        correct += (predicted == answers).sum().item()

                        all_predicted.extend(predicted.cpu().numpy())
                        all_answers.extend(answers.cpu().numpy())
                    
                    #Compute metrics (accuracy, precision, recall, f_score)
                    accuracy = 100 * correct / total
                    precision = precision_score(all_answers, all_predicted)
                    recall = recall_score(all_answers, all_predicted)
                    f_score = f1_score(all_answers, all_predicted)

                    if f_score > best_val_fscore: #Save the model with the best F-score
                        best_val_fscore = f_score
                        torch.save(model.state_dict(), f'{experiment_name}.pth')
                    print(f'Iteration {i+1}, Validation Loss: {loss.item()}, Accuracy: {accuracy}%, Precision: {precision}, Recall: {recall}, F-score: {f_score}')
                    accuracy_results[f'Epoch {epoch+1}'] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F-score': f_score}  

    # Save the accuracy, precision, and recall results to a JSON file
    with open(f'results_{experiment_name}.json', 'w') as f:
        json.dump(accuracy_results, f)

#VALIDATION RESULTS 
#This loop can be used to quickly obtain results in the valid set by loading pretrained models. It can be used
#to reproduce our validation experiment results, but models must be trained first.
elif args.train == False and args.mode == "valid":
    # Load the trained model
    if args.bestmethod == False: 
        model_path = f'{experiment_name}.pth'
    elif args.bestmethod == True and args.pretrained == True: #For evaluating the best (final) method, it gets loaded directly from the path in BCV002. One could also train the model with the best method and then evaluate it.
        model_path = "/home/nandradev/ProyectoAML/Metodo/ProyectoAML/CLIP_ViT-B32_concat_nl1_hd1024_BCELoss_w40.pth"
    elif args.bestmethod == True and args.pretrained == False:
        model_path = 'CLIP_ViT-B32_concat_nl1_hd1024_BCELoss_w40.pth' #Loads pretrained final model from the current directory, requires training it first.
    model.load_state_dict(torch.load(model_path))

    all_predicted = []
    all_answers = []
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            images = batch["image"].to(device).half()
            questions = batch["question"].squeeze(1).to(device)
            answers = batch["answer"].unsqueeze(1).to(device).float()
            print(f'Evaluated batches: {i+1}/{len(valid_loader)}')
            outputs = model(images, questions)

            all_predicted.extend(outputs.cpu().numpy())
            all_answers.extend(answers.cpu().numpy())

    all_predicted = [item.tolist() for sublist in all_predicted for item in sublist]
    all_answers = [item.tolist() for sublist in all_answers for item in sublist]

    results = {'Predicted': all_predicted, 'Labels': all_answers}
    with open(f'predictions_{experiment_name}.json', 'w') as f: #We store predictions and labels for the PR curve
        json.dump(results, f)
    
    # Calculate precision, recall, and F-score for a threshold of 0.5
    predicted_05 = (np.array(all_predicted) >= 0.5).astype(float)
    precision_05 = precision_score(all_answers, predicted_05)
    recall_05 = recall_score(all_answers, predicted_05)
    fscore_05 = f1_score(all_answers, predicted_05)

    #With the 0.5 threshold we compute metrics reported in the paper for the validation set
    results_05 = {'Precision': precision_05, 'Recall': recall_05, 'F-score': fscore_05}

    #Save the 0.5 threshold results to a JSON file
    with open(f'05_results_{experiment_name}.json', 'w') as f:
        json.dump(results_05, f)

    #Obtain precision-recall curve
    precisions, recalls, _ = precision_recall_curve(all_answers, all_predicted)

    #Compute AUC of the precision-recall curve
    auc_pr = auc(recalls, precisions)
    print(f'Area under the curve: {auc_pr},Precision: {precision_05}, Recall: {recall_05}, F-score: {fscore_05}')

    #Plot the precision-recall curve and save it
    plt.plot(recalls, precisions, linestyle='-', label=f'{experiment_name}')
    plt.grid()
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve {experiment_name}')
    plt.savefig(f"PRCurve{experiment_name}.jpg")


#TEST RESULTS
elif args.train == False and args.mode == "test":

    #Create dataloader for test images
    print("Processing test images...")
    resized_filtered_test = Dataset.from_dict({
        'image':[image_transform(image) for image in tqdm(filtered_test['image'])],
        'question':[process_question(question) for question in tqdm(filtered_test['question'])],
        'answer': [process_answer(answer) for answer in filtered_test['answer']],
    })
    test_loader = DataLoader(resized_filtered_test,batch_size=args.valid_batch_size, collate_fn=collate_fn)

    # Load the trained model
    if args.bestmethod == False: 
        model_path = f'{experiment_name}.pth'
    elif args.bestmethod == True and args.pretrained == True: #For evaluating the best (final) method, it gets loaded directly from the path in BCV002. One could also train the model with the best method and then evaluate it.
        model_path = "/home/nandradev/ProyectoAML/Metodo/ProyectoAML/CLIP_ViT-B32_concat_nl1_hd1024_BCELoss_w40.pth"
    elif args.bestmethod == True and args.pretrained == False:
        model_path = 'CLIP_ViT-B32_concat_nl1_hd1024_BCELoss_w40.pth' #Loads pretrained final model from the current directory, requires training it first.

    model.load_state_dict(torch.load(model_path))
    all_images = []
    all_predicted = []
    all_answers = []
    all_questions = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["image"].to(device).half()

            questions = batch["question"].squeeze(1).to(device)
            answers = batch["answer"].unsqueeze(1).to(device).float()
            print(f'Evaluated batches: {i+1}/{len(test_loader)}')
            outputs = model(images, questions)
            all_questions.extend(questions.cpu().numpy())
            all_images.extend(images.cpu().numpy())
            all_predicted.extend(outputs.cpu().numpy())
            all_answers.extend(answers.cpu().numpy())

    all_predicted = [item.tolist() for sublist in all_predicted for item in sublist]
    all_answers = [item.tolist() for sublist in all_answers for item in sublist]

    results = {'Predicted': all_predicted, 'Labels': all_answers}
    with open(f'predictions_{experiment_name}-Test.json', 'w') as f:
        json.dump(results, f)
    
    #Calculate precision, recall, and F-score for threshold of 0.5
    predicted_05 = (np.array(all_predicted) >= 0.5).astype(float)
    precision_05 = precision_score(all_answers, predicted_05)
    recall_05 = recall_score(all_answers, predicted_05)
    fscore_05 = f1_score(all_answers, predicted_05)

    results_05 = {'Precision': precision_05, 'Recall': recall_05, 'F-score': fscore_05}

    #Save the results to a separate JSON file
    with open(f'05_results_{experiment_name}-Test.json', 'w') as f:
        json.dump(results_05, f)

    #Compute precision-recall curve
    precisions, recalls, _ = precision_recall_curve(all_answers, all_predicted)

    #Compute AUC of the precision-recall curve, also print results. These should be the same as the final method results in the paper.
    auc_pr = auc(recalls, precisions)
    print(f'Area under the curve: {auc_pr},Precision: {precision_05}, Recall: {recall_05}, F-score: {fscore_05}')

    #Plot the precision-recall curve for the final method
    plt.plot(recalls, precisions, linestyle='-', label=f'{experiment_name}')
    plt.grid()
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Test {experiment_name}')
    plt.savefig(f"PRCurve{experiment_name}-Test.jpg")

    all_questions = np.array(all_questions)
    all_images = np.array(all_images)
    all_predicted = np.array(all_predicted)
    all_answers = np.array(all_answers)


    #Cualitative results for the final method:
    os.makedirs('wrong_high_confidence_images', exist_ok=True)
    os.makedirs('wrong_low_confidence_images', exist_ok=True)
    #Get indices of instances where the model was wrong and the label is 0 (no)
    wrong_high_confidence_indices = np.where((all_predicted.round() != all_answers) & (all_answers == 0))

    #Sort these indices by prediction score, we want the highest scores
    sorted_indices = wrong_high_confidence_indices[0][np.argsort(all_predicted[wrong_high_confidence_indices])]
    
    quali_results = {}

    #Save the top 10 images where the model was wrong and had high confidence
    for i in range(10):
        img = Image.fromarray((all_images[sorted_indices[-i-1]].transpose((1, 2, 0)) * 255).astype(np.uint8))
        img_name = f'img_{i+1}.png'
        img.save(f'wrong_high_confidence_images/{img_name}')
        score = all_predicted[sorted_indices[-i-1]].tolist()
        question = all_questions[sorted_indices[-i-1]].tolist()
        truequestion = filtered_test['question'][sorted_indices[-i-1]]
        quali_results[img_name] = {'score': score, 'truequestion': truequestion}
   
    #Get indices of instances where the model was wrong and the label is 1 (yes)
    wrong_low_confidence_indices = np.where((all_predicted.round() != all_answers) & (all_answers == 1))

    #Sort these indices by prediction score (we want lowest scores)
    sorted_indices_low = wrong_low_confidence_indices[0][np.argsort(all_predicted[wrong_low_confidence_indices])]

    #Save the top 10 images where the model was wrong and had low confidence (predicted no)
    for i in range(10):
        img = Image.fromarray((all_images[sorted_indices_low[i]].transpose((1, 2, 0)) * 255).astype(np.uint8))
        img_name = f'img_{i+11}.png'
        img.save(f'wrong_low_confidence_images/{img_name}')
        score = all_predicted[sorted_indices_low[i]].tolist()
        question = all_questions[sorted_indices_low[i]].tolist()
        truequestion = filtered_test['question'][sorted_indices_low[i]]
        quali_results[img_name] = {'score': score, 'truequestion': truequestion}

    # Save the results to a JSON file
    with open('Qualitresults-Finalmethod.json', 'w') as f:
        json.dump(quali_results, f)

#DEMO CODE
if args.mode == 'demo' and args.img != 'None':
    print("Loading model and test image...")
    model = CustomCLIPModel(clip_model).to(device)
    #We only use the demo with the final method, so we load the pretrained model directly from the path in BCV002.
    model_path = "/home/nandradev/ProyectoAML/Metodo/ProyectoAML/CLIP_ViT-B32_concat_nl1_hd1024_BCELoss_w40.pth"
    model.load_state_dict(torch.load(model_path))
    img = Image.open(args.img)
    img = image_transform(img).unsqueeze(0)
    question = input("Enter a question: ")
    question = process_question(question)
        
    with torch.no_grad():
        output = model(img.to(device).half(), question.to(device))
        print("The answer is yes" if output > 0.5 else "The answer is no")
