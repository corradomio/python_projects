# Import the necessary modules
from skllm.datasets import get_classification_dataset
from skllm.config import SKLLMConfig
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier

# Configure the credentials
SKLLMConfig.set_openai_key("<YOUR_KEY>")
SKLLMConfig.set_openai_org("<YOUR_ORGANIZATION_ID>")

# Load a demo dataset
X, y = get_classification_dataset() # labels: positive, negative, neutral

# Initialize the model and make the predictions
clf = ZeroShotGPTClassifier(model="gpt-4")
clf.fit(X,y)
clf.predict(X)
