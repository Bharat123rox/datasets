# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os

import datasets


_CITATION = """
@inproceedings{wei-etal-2018-airdialogue,
    title = "{A}ir{D}ialogue: An Environment for Goal-Oriented Dialogue Research",
    author = "Wei, Wei  and
      Le, Quoc  and
      Dai, Andrew  and
      Li, Jia",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1419",
    doi = "10.18653/v1/D18-1419",
    pages = "3844--3854",
}
"""

_DESCRIPTION = """\
AirDialogue is a benchmark dataset for goal-oriented dialogue generation research.
"""


_HOMEPAGE = "https://github.com/google/airdialogue"


_LICENSE = "Apache License 2.0"


_URL = "https://storage.googleapis.com/airdialogue/airdialogue_data.tar.gz"



class AirDialogue(datasets.GeneratorBasedBuilder):
    """Dataset for goal-oriented dialogue generation research"""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="data", description="The dialogues between the customer and agent and the expected action or outcome"),
        datasets.BuilderConfig(name="kb", description="The knowledge base for the dialogues and conversations"),
    ]

    DEFAULT_CONFIG_NAME = "data"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        if self.config.name == "kb":
            features = datasets.Features(
                {
                    "kb": datasets.features.Sequence(
                        {
                            "return_airport": datasets.Value("string"),
                            "airline": datasets.Value("string"),
                            "departure_day": datasets.Value("string"),
                            "departure_airport": datasets.Value("string"),
                            "flight_number": datasets.Value("int32"),
                            "departure_month": datasets.Value("string"),
                            "departure_time_num": datasets.Value("int32"),
                            "class": datasets.features.ClassLabel(names=["economy", "business"]),
                            "return_time_num": datasets.Value("int32"),
                            "return_month": datasets.Value("string"),
                            "return_day": datasets.Value("string"),
                            "num_connections": datasets.features.ClassLabel(names=["0", "1", "many"]),
                            "price": datasets.Value("int32")

                        }
                    ),
                    "reservation": datasets.Value("int32")
                }
            )
        else:
            features = datasets.Features(
                {
                    "search_info": datasets.features.Sequence(
                        {
                            "timestmamp": datasets.Value("int32"),
                            "button_name": datasets.Value("string"),
                            "field_name": datasets.Value("string"),
                            "field_value": datasets.Value("string")
                        }
                    ),
                    "action": datasets.features.Sequence(
                        {
                            "status": datasets.features.ClassLabel(names=["book", "no_flight", "change", "no_reservation", "cancel"]),
                            "name": datasets.Value("string"),
                            "flight": datasets.Sequence(datasets.Value("int32"))
                        }
                    ),
                    "intent": datasets.features.Sequence(
                        {
                            "return_month": datasets.Value("string"),
                            "return_day": datasets.Value("int32"),
                            "max_price": datasets.Value("int32"),
                            "departure_airport": datasets.Value("string"),
                            "departure_time": datasets.Value("string"),
                            "max_connections": datasets.features.ClassLabel(names=["0", "1", "many"]),
                            "departure_day": datasets.Value("string"),
                            "goal": datasets.features.ClassLabel(names=["book", "change", "cancel"]),
                            "departure_month": datasets.Value("string"),
                            "name": datasets.Value("string"),
                            "return_airport": datasets.Value("string")
                        }
                    ),
                    "timestamps": datasets.Sequence(datasets.Value("int32")),
                    "dialogue": datasets.Sequence(datasets.Value("string")),
                    "expected_action": datasets.features.Sequence(
                        {
                            "status": datasets.features.ClassLabel(names=["book", "no_flight", "change", "no_reservation", "cancel"]),
                            "name": datasets.Value("string"),
                            "flight": datasets.Sequence(datasets.Value("int32"))
                        }
                    ),
                    "correct_sample": datasets.Value("bool_")
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=("dialogue", "expected_action") if self.config.name == "kb" else None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive 
        url = _URL
        extracted_dir = dl_manager.download_and_extract(url)
        data_dir = os.path.join(extracted_dir, "airdialogue_data", "airdialogue")
        TRAIN = "train_{config}.json"
        DEV = "dev_{config}.json"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, TRAIN.format(config=self.config.name)),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, DEV.format(config=self.config.name)),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """

        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                print(row)
                data = json.loads(row)
                if self.config.name == "kb":
                    yield id_, {
                        "kb": data["kb"],
                        "reservation": data["reservation"],
                    }
                else:
                    if "search_info" in data:
                        calculated_search_info = []
                        for item in data["search_info"]:
                            calculated_search_info.append({
                                "timestmamp": item["timestmamp"],
                                "button_name": item.get("button_name", ""),
                                "field_name": item.get("field_name", ""),
                                "field_value": item.get("field_value", "")
                            })
                    if data["correct_sample"]:
                        yield id_, {
                            "search_info": calculated_search_info,
                            "action": {
                                "status": data["action"].get("status", "book"),
                                "name": data["action"].get("name", ""),
                                "flight": data["action"].get("flight", [])
                            },
                            "intent": {
                                "return_month": data["intent"].get("return_month", ""),
                                "return_day": data["intent"].get("return_day", ""),
                                "max_price": data["intent"].get("max_price", ""),
                                "departure_airport": data["intent"].get("departure_airport", ""),
                                "departure_time": data["intent"].get("departure_time", ""),
                                "max_connections": data["intent"].get("max_connections", ""),
                                "departure_day": data["intent"].get("departure_day", ""),
                                "goal": data["intent"].get("goal", ""),
                                "departure_month": data["intent"].get("departure_month", ""),
                                "name": data["intent"].get("name", ""),
                                "return_airport": data["intent"].get("return_airport", "")
                            },
                            "timestamps": data["timestamps"],
                            "dialogue": data["dialogue"],
                            "expected_action": {
                                "status": data["expected_action"].get("status", "book"),
                                "name": data["expected_action"].get("name", ""),
                                "flight": data["expected_action"].get("flight", [])
                            },
                            "correct_sample": data["correct_sample"]
                        }

