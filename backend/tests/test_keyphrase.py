import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'AI')))
import modelTokenizerHandler



phrase = "a slice of life anime with a lot of comedy and some romance, like 'Komi-san wa Komyushou Desu' or 'Kaguya-sama: Love is War'. I also like action series like 'Attack on Titan' or 'Jujutsu Kaisen'."

print(modelTokenizerHandler.extract_keyphrases(phrase))


