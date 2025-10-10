"""Generic detector."""
import os
import pickle

import torch

from external.adaptors import yolox_adaptor


class Detector(torch.nn.Module):
    K_MODELS = {"yolox"}

    def __init__(self, model_type, path, dataset):
        super().__init__()
        if model_type not in self.K_MODELS:
            raise RuntimeError(f"{model_type} detector not supported")

        self.model_type = model_type
        self.path = path
        self.dataset = dataset
        self.model = None

        os.makedirs("./cache", exist_ok=True)
        self.cache_path = os.path.join(
            "./cache", f"det_{os.path.basename(path).split('.')[0]}.pkl"
        )
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as fp:
                self.cache = pickle.load(fp)
        else:
            self.initialize_model()

    def initialize_model(self):
        """Wait until needed."""
        if self.model_type == "yolox":
            self.model = yolox_adaptor.get_model(self.path, self.dataset)

    def forward(self, batch, tag=None):
        """If tag is a list (batch tags), cache per item; otherwise cache single item."""
        if isinstance(tag, (list, tuple)):
            # Attempt to serve entirely from cache if all present
            cached_outputs = []
            all_cached = True
            for t in tag:
                if t in self.cache:
                    cached_outputs.append(self.cache[t])
                else:
                    all_cached = False
                    break
            if all_cached:
                return cached_outputs

        else:
            if tag in self.cache:
                return self.cache[tag]

        if self.model is None:
            self.initialize_model()

        with torch.no_grad():
            batch = batch.half()
            output = self.model(batch)

        # Persist to cache
        if isinstance(tag, (list, tuple)):
            # output is a list, align lengths
            if isinstance(output, (list, tuple)) and len(output) == len(tag):
                for t, o in zip(tag, output):
                    if o is not None:
                        self.cache[t] = o.cpu().detach()
            return output
        else:
            if output is not None:
                # For single item forward, yolox_adaptor returns list of length 1
                if isinstance(output, (list, tuple)):
                    o0 = output[0]
                else:
                    o0 = output
                if o0 is not None:
                    self.cache[tag] = o0.cpu().detach()
                return o0

        return output

    def dump_cache(self):
        with open(self.cache_path, "wb") as fp:
            pickle.dump(self.cache, fp)
