
Setup python visual studio environment [with typing](https://code.visualstudio.com/docs/python/tutorial-fastapi):

Open the Settings editor (⌘,).
Search for "python type checking mode" and set it to basic for basic type checking. Pylance will now show diagnostics and warnings to catch simple type-related errors. Alternatively, you can set it to strict to enforce more advanced type checking rules.
Next, search for "Python inlay type hints", and enable inlay hints for "Variable Types" and "Function Return Types"

# Install environment
```
conda create -n rag python
conda activate rag
conda install ninja
conda install pytorch -c pytorch
conda install -c pytorch faiss-cpu=1.8.0
#conda install -c pytorch -c nvidia faiss-gpu=1.8.0
conda install -c conda-forge pybind11

pip install colbert-ai srsly requests
#pip install ragatouille pyvespa requests
# upgrade to latest llama-index for SemanticSplitter
pip install llama-index --upgrade
#pip install llama-index-readers-web

# for docx import
pip install docx2txt
#https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface.html
pip install llama-index-embeddings-huggingface
#pip install llama-index-embeddings-instructor
#pip install transformers optimum[exporters]
pip install 'huggingface-hub==0.22.2'
pip install 'transformers==4.39.3'
pip install dspy-ai

pip install aiosqlite
pip install fastapi
pip install "uvicorn[standard]"

conda env export --no-build > environment.yml

python ingest.py

conda deactivate 
conda remove --name rag --all
```

In visual studio code select conda environment 
⇧⌘P python: s

# Run

Run FastAPI server

in dev
```
uvicorn main:app --reload
```

in prod
```
uvicorn main:app
```

# Build docker
ECR_REPOSITORY_NAME=rag-cm
ECR_AWS_REGION=us-west-2

VERSION=$( cat VERSION )
echo "export BUILD_VERSION=${VERSION}" >> $BASH_ENV
echo 'export FULL_REPOSITORY_NAME="${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}"' >> $BASH_ENV
echo 'export IMAGE_TAGGED_VERSION_BUILD="${FULL_REPOSITORY_NAME}:${BUILD_VERSION}-${CIRCLE_BUILD_NUM}"' >> $BASH_ENV
echo 'export IMAGE_TAGGED_VERSION="${FULL_REPOSITORY_NAME}:${BUILD_VERSION}"' >> $BASH_ENV
echo 'export IMAGE_TAGGED_LATEST="${FULL_REPOSITORY_NAME}:latest"' >> $BASH_ENV


docker build -t $IMAGE_TAGGED_VERSION_BUILD -t $IMAGE_TAGGED_VERSION -t $IMAGE_TAGGED_LATEST --build-arg IMG_VERSION=$BUILD_VERSION --build-arg IMG_BUILD_NUMBER=$CIRCLE_BUILD_NUM .

docker run -it --entrypoint=/bin/bash 028211419619.dkr.ecr.us-west-2.amazonaws.com/rag-cm

# Known issues that need to be fixed in the ColBERT dependency

## Semaphore leak
When shutting down the server we get this
```
miniconda/base/envs/rag/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```

caused by use of tqdm.tqdm() in

colbert/search/index_loader.py -> _load_doclens
colbert/indexing/codecs/residual_embeddings.py -> load_chunks

might be related to this:
https://github.com/tqdm/tqdm/issues/1321

## ColBERT creating Checkpoint objects in multiple places
ColBERT creates a `Checkpoint` object in `Searcher`, `Indexer` and parent class `CollectionIndexer`, `IndexUpdater`

## ColBERT not keeping different parts in sync
When we have a searcher loaded and rebuild the index through `Indexer`, the searcher does not know anything about the added components. We currently have to trigger a reload of the searcher to make this work but it would be better if we could avoid this.
Details:
When indexing, `Indexer` through `CollectionIndexer` calculated `doclens`
`Searcher` stores its own `IndexScorer(IndexLoader)` in `self.ranker` that is not updated when the index is updated. Its `self.doclens` is only loaded at initialization. 

## No control over the passage ids inside ColBERT
ColBERT assigns passage ids in sequential order. We can't insert our own ids when building the index. And when adding elements, it calculates the next ids based on doclens. The problem is that when you delete elements from the index and the insert more elements to it, self.doclens wil return the amount of currently active passages (excluding the deleted ones). But the indexes in sql still have holes in them. When we now generate new ids following what ColBERT wants to use, then these ids are already used in the database.

## When creating a `Searcher`, we can't supply an index path
Instead, the constructor of `Searcher` requires separate `index_root` and `index` to be supplied. But internally, it's just joined into a single path. But because of that, our system cannot just have a single path variable and we will always need to carry both individual values. Instead, we'd rather use `self.config.index_path`. The same goes for `Indexer.index` which expects `index_name=name`

## State is stored in config objects that should be configurable for each call
E.g. we use `_upgrade_searcher_maxlen` to re-configure the searcher for each query. This doesn't work if there are multiple parallel queries.Also `self.searcher.configure(ncells=min((k // 32 + 2), base_ncells))` and `self.searcher.configure(ndocs=max(k * 4, base_ndocs)` cannot be set per query

# Colbert code learnings

Creating 128 partitions/centroids
146 embeddings/len(emb2pid) = 146

self.ivf = 146
self.ivf_lengths = 128 (number of centroids)

## "codec" storage
`centroids.pt` embeddings for centroids (128 centroids x 128 dims float16)
`avg_residual.pt` contains just one number: avg_residual = 0.010149186477065086
`buckets.pt` contains `bucket_cutoffs` [255 floats], `bucket_weights` [256 floats]

`metadata.json`: num_chunks=1, num_partitions (centroids) = 128, num_embeddings = 146, avg_doclen= 11.23076923076923
`plan.json`: num_embeddings_est = 145.99999904632568, avg_doclen_est = 11.230769157409668

## passage id storage
`ivf.pid.pt` contains `ivf`, `ivf_lengths`

`ivf` stores for each centroid the passage ids that belong to it. Index is the offsets based on the lengths in ivf_lengths. If a passage has multiple tokens, each token might belong to a different centroid. That's why len(ivf) is the same as number of embeddings
`ivf_lengths` stores for each centroid how many pids it stores. Can be 0 if there's no pids

`{chunk_idx}.metadata.json`:
each chunk stores passage ids starting from "passage_offset"  to passage_offset + num_passages

`{chunk_idx}.metadata.json` chunk meta data should be moved into into SQL

## Embedding ("codes") storage:
`doclens.{chunk_idx}.json` stores the number of embeddings per passage. Index = passage id. For deleted passage ids, len = 0. When loaded in the searcher, all all combined in `self.searcher.ranker.doclens`
NOTE: Adding all embedding counts together defines index/id of next embedding to be stored. This is usually stored in a dynamically created `offset`. The number of all lengths seems to be same as `len(self.ivf)`???

`{chunk_idx}.codes.pt` (= 146) -> content = index of the centroid the embedding belongs to (int32) (min value 0, max value 127)  -> index=relative embedding id (from doclen sum)
`{chunk_idx}.residuals.pt` ( = [146, 128]) -> content = delta to the cetroid on each of the 128 dims (uint8) (min value 0, max value 255) index=relative embedding id (from doclen sum per)


# Code improvements
`segmented_lookup.cpp` / `StridedTensor.segmented_lookup` passes in `pids` but it's just needed for the number of pids. And that can also be read by looking at the size of `offsets` or `lengths` arrays