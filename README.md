# Just a fork of: https://github.com/AllAboutAI-YT/easy-local-rag.git
## did some modifications including saving the embedings and some PE


### Run with generate and save embeddings
python3 localrag.py --model llama3.2 --embedding-model mxbai-embed-large --device cpu --save-embeddings --embedding-file embeddings.pt

### Run with existing embeddings:
python3 localrag.py --model llama3.2 --embedding-model mxbai-embed-large --device cpu --use-cache --embedding-file embeddings.pt


