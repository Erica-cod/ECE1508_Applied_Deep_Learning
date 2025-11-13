(base) kevinlin@nuke:~/utece/ECE1508_Applied_Deep_Learning$ python scripts/finetune_distilbert_recipe_mpr.py
2025-11-13 00:00:58 - INFO - __main__ - Loading model: distilbert-base-uncased
2025-11-13 00:00:58 - INFO - __main__ - Output directory: /home/kevinlin/models/hub/distilbert-finetuned-recipe-mpr
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48.0/48.0 [00:00<00:00, 680kB/s]
config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 483/483 [00:00<00:00, 7.37MB/s]
vocab.txt: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 6.17MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 466k/466k [00:00<00:00, 4.90MB/s]
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 268M/268M [00:12<00:00, 22.3MB/s]
Some weights of DistilBertForMultipleChoice were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
2025-11-13 00:01:11 - INFO - __main__ - Loaded 500 examples from data/500QA.json
2025-11-13 00:01:11 - INFO - __main__ - Train size: 450, Eval size: 50
2025-11-13 00:01:11 - INFO - __main__ - Tokenizing dataset...
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [00:00<00:00, 1763.13 examples/s]
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 3067.94 examples/s]
/home/kevinlin/utece/ECE1508_Applied_Deep_Learning/scripts/finetune_distilbert_recipe_mpr.py:387: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
2025-11-13 00:01:15 - INFO - __main__ - Starting training...
{'loss': 1.6104, 'grad_norm': 1.3067591190338135, 'learning_rate': 2.45e-05, 'epoch': 0.88}                                                                                                                                                    
{'loss': 1.5853, 'grad_norm': 4.794619560241699, 'learning_rate': 4.9500000000000004e-05, 'epoch': 1.75}                                                                                                                                       
{'loss': 1.4023, 'grad_norm': 14.32573127746582, 'learning_rate': 1.5492957746478872e-05, 'epoch': 2.63}                                                                                                                                       
{'train_runtime': 15.9782, 'train_samples_per_second': 84.49, 'train_steps_per_second': 10.702, 'train_loss': 1.4894540574815538, 'epoch': 3.0}                                                                                                
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 171/171 [00:15<00:00, 10.71it/s]
***** train metrics *****
  epoch                    =        3.0
  total_flos               =   416365GF
  train_loss               =     1.4895
  train_runtime            = 0:00:15.97
  train_samples_per_second =      84.49
  train_steps_per_second   =     10.702
2025-11-13 00:01:31 - INFO - __main__ - Running final evaluation...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 27.66it/s]
***** eval metrics *****
  epoch                   =        3.0
  eval_accuracy           =       0.28
  eval_loss               =     1.6373
  eval_runtime            = 0:00:00.30
  eval_samples_per_second =    163.151
  eval_steps_per_second   =     22.841
2025-11-13 00:01:32 - INFO - __main__ - Saving fine-tuned model to /home/kevinlin/models/hub/distilbert-finetuned-recipe-mpr
2025-11-13 00:01:32 - INFO - __main__ - Training complete!
2025-11-13 00:01:32 - INFO - __main__ - Final evaluation accuracy: 0.2800
(base) kevinlin@nuke:~/utece/ECE1508_Applied_Deep_Learning$ python scripts/evaluate_distilbert_recipe_mpr.py \
    --model-path ~/models/hub/distilbert-finetuned-recipe-mpr
2025-11-13 00:01:38 - INFO - Loading model from: /home/kevinlin/models/hub/distilbert-finetuned-recipe-mpr
2025-11-13 00:01:38 - INFO - Using device: cuda
2025-11-13 00:01:38 - INFO - Loaded 500 examples from data/500QA.json
2025-11-13 00:01:38 - INFO - Evaluating on 500 examples...
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:02<00:00, 167.79it/s]

================================================================================
RECIPE-MPR EVALUATION RESULTS
================================================================================

Overall Performance:
  Accuracy: 69.40% (347/500)
  ✗ Below goal by 0.60% (target: 70.0%)

Accuracy by Query Type:
  Type            Accuracy     Count     
  ----------------------------------------
  Specific         74.83%      151       
  Analogical       73.33%      30        
  Negated          69.72%      109       
  Commonsense      68.66%      268       
  Temporal         62.50%      32        

Example Predictions:

Correct Predictions (showing 5):

  1. Question: Can I get a soup recipe that contains chicken or beef and other ingredients that aren't vegetables?...
     Answer: Assorted cheese soup with chicken
     Query types: Commonsense, Negated

  2. Question: I'd like to make a classic French fruit tart but I don't like raspberries...
     Answer: French fruit tart with grapes, kiwi, and strawberries
     Query types: Specific, Negated

  3. Question: Can I have a recipe for barbecue meat, but not seafood?...
     Answer: Beef short ribs cooked with barbecue sauce
     Query types: Specific, Negated

  4. Question: My kids want to use the new popsicle maker I bought...
     Answer: Popsicles made from yoghurt, apple juice and assorted fruit
     Query types: 

  5. Question: How do I cook an Italian meatball soup?...
     Answer: Italian meatball and pasta with cheese, cooked in soup stew
     Query types: Specific

Incorrect Predictions (showing 5):

  1. Question: Today's really hot and I'm craving for a peach flavoured treat...
     Predicted: A jar of peaches covered in peach wine and syrup
     Correct:   Gelato made from peach, heavy cream, egg yolks, truvia sweetener, and plain yogu
     Query types: Specific

  2. Question: How can I eat caviar with crackers?...
     Predicted: Potatoes stuffed with caviar
     Correct:   Caviar dip from caviar and sour cream, nestled in ice
     Query types: Commonsense

  3. Question: My family wants a recipe for peach cobbler that's healthier...
     Predicted: Cobbler dessert made using three types of berries
     Correct:   Peach cobbler dessert containing whole wheat flour
     Query types: Specific, Commonsense

  4. Question: I'm sick and tired of a traditional breakfast, would I be able to have a unique breakfast dish witho...
     Predicted: Bacon
     Correct:   Low fat strawberry parfait with apricot, yogurt, raisins, and almonds
     Query types: Negated

  5. Question: What's a noodle dish I can make with a creamy sauce?...
     Predicted: Spaghetti noodles with marinara sauce made with olive oil, garlic, tomatoes, sal
     Correct:   Spaghetti with mushrooms, onion, green pepper, chicken breasts, and alfredo sauc
     Query types: 

================================================================================