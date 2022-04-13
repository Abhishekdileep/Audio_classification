import hub
import tensorflow_datasets as tdfs
ds = hub.load("hub://activeloop/speech-commands-test")
print(ds)