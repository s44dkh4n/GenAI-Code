from langchain_mistralai import MistralAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

load_dotenv()

embeddings = MistralAIEmbeddings(model="mistral-embed")

text = """The James Webb Space Telescope has revolutionized our understanding of the early universe by capturing infrared light from galaxies formed over 13 billion years ago. Equipped with a 6.5-meter primary mirror and highly sensitive spectrometers, the observatory can pierce through thick clouds of cosmic dust to reveal stellar nurseries. Scientists use this data to study the atmospheric composition of distant exoplanets, searching for chemical signatures like water vapor, carbon dioxide, and methane that might indicate habitability. Meanwhile, in the deepest trenches of Earth's oceans, marine biologists are discovering entirely new ecosystems thriving in complete darkness. Hydrothermal vents on the ocean floor release superheated, mineral-rich water that sustains unique communities of tube worms, ghost crabs, and chemosynthetic bacteria. Unlike terrestrial life forms that rely on sunlight and photosynthesis, these deep-sea organisms convert toxic chemicals into energy, offering vital clues about how life might evolve on icy moons like Europa or Enceladus. Transitioning from natural wonders to global infrastructure, the rapid expansion of renewable energy systems is reshaping international power grids. Continuous drops in the manufacturing costs of photovoltaic solar panels and utility-scale wind turbines have made clean energy cheaper than fossil fuels in many parts of the world. However, integrating these intermittent power sources requires massive investments in next-generation grid energy storage, such as solid-state batteries and pumped-storage hydropower. At the same time, the software driving modern infrastructure is undergoing a massive transformation due to generative artificial intelligence. Large language models are transitionining from simple chatbots into autonomous agents capable of writing code, debugging complex systems, and managing supply chains. This shift raises critical ethical questions regarding data privacy, structural employment changes, and the urgent need for robust alignment frameworks to keep AI systems safe."""

splitter = SemanticChunker(
    embeddings= embeddings,
    breakpoint_threshold_type= "percentile",
    breakpoint_threshold_amount= 70
)

chunks = splitter.split_text(text= text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} \n{chunk} \n{'=='*50}\n")
