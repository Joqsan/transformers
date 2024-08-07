from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Create a custom logger
import logging 
logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('logs/t5_original.log', "w")

# Create formatters and add them to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Set the logging level
logger.setLevel(logging.DEBUG)  # or any other level you prefer

# Get the transformers logger
transformers_logger = logging.getLogger("transformers")

# Add the file handler to the transformers logger
transformers_logger.addHandler(f_handler)

# Set the logging level for the transformers logger
transformers_logger.setLevel(logging.INFO)  # or any other level you prefer


tokenizer = AutoTokenizer.from_pretrained("ai-forever/sage-fredt5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/sage-fredt5-large", device_map='cpu')

sentence = "И не чсно прохожим в этот день непогожйи почему я веселый такйо"
inputs = tokenizer(sentence, max_length=None, padding="longest", truncation=False, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_length = inputs["input_ids"].size(1) * 1.5)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))