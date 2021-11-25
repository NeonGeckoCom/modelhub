import requests
from json_database import JsonStorageXDG

MANIFEST = JsonStorageXDG("manifest", subfolder="ModelZoo")
MANIFEST_URL = "https://github.com/NeonJarbas/modelhub/raw/models/models/manifest.json"


def populate_manifest():
    # download manifest file from server
    try:
        r = requests.get(MANIFEST_URL).json()
        MANIFEST.update(r)
        MANIFEST.store()
    except:
        # model urls, used only if manifest does not yet exist
        # manually maintained list !
        DEFAULT_MANIFEST = {
            'nltk_brown_brill_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_brown_brill_tagger.pkl',
            'nltk_brown_dtree_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_brown_dtree_tagger.pkl',
            'nltk_brown_maxent_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_brown_maxent_tagger.pkl',
            'nltk_brown_ngram_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_brown_ngram_tagger.pkl',
            'nltk_brown_perceptron_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_brown_perceptron_tagger.pkl',
            'nltk_brown_tnt_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_brown_tnt_tagger.pkl',
            'nltk_cess_cat_brill_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_cess_cat_brill_tagger.pkl',
            'nltk_cess_cat_ngram_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_cess_cat_ngram_tagger.pkl',
            'nltk_cess_cat_udep_brill_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_cess_cat_udep_brill_tagger.pkl',
            'nltk_cess_cat_udep_ngram_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_cess_cat_udep_ngram_tagger.pkl',
            'nltk_cess_cat_udep_perceptron_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_cess_cat_udep_perceptron_tagger.pkl',
            'nltk_cess_esp_brill_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_cess_esp_brill_tagger.pkl',
            'nltk_cess_esp_ngram_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_cess_esp_ngram_tagger.pkl',
            'nltk_cess_esp_perceptron_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_cess_esp_perceptron_tagger.pkl',
            'nltk_floresta_brill_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_floresta_brill_tagger.pkl',
            'nltk_floresta_macmorpho_brill_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_floresta_macmorpho_brill_tagger.pkl',
            'nltk_floresta_macmorpho_ngram_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_floresta_macmorpho_ngram_tagger.pkl',
            'nltk_floresta_macmorpho_perceptron_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_floresta_macmorpho_perceptron_tagger.pkl',
            'nltk_floresta_ngram_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_floresta_ngram_tagger.pkl',
            'nltk_floresta_perceptron_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_floresta_perceptron_tagger.pkl',
            'nltk_floresta_tnt_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_floresta_tnt_tagger.pkl',
            'nltk_macmorpho_brill_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_macmorpho_brill_tagger.pkl',
            'nltk_macmorpho_ngram_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_macmorpho_ngram_tagger.pkl',
            'nltk_macmorpho_perceptron_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_macmorpho_perceptron_tagger.pkl',
            'nltk_macmorpho_udep_brill_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_macmorpho_udep_brill_tagger.pkl',
            'nltk_macmorpho_udep_ngram_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_macmorpho_udep_ngram_tagger.pkl',
            'nltk_macmorpho_udep_perceptron_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_macmorpho_udep_perceptron_tagger.pkl',
            'nltk_nilc_brill_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_nilc_brill_tagger.pkl',
            'nltk_nilc_dtree_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_nilc_dtree_tagger.pkl',
            'nltk_nilc_ngram_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_nilc_ngram_tagger.pkl',
            'nltk_nilc_perceptron_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_nilc_perceptron_tagger.pkl',
            'nltk_onto5_ngram_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_onto5_ngram_tagger.pkl',
            'nltk_onto5_perceptron_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_onto5_perceptron_tagger.pkl',
            'nltk_treebank_brill_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_treebank_brill_tagger.pkl',
            'nltk_treebank_hmm_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_treebank_hmm_tagger.pkl',
            'nltk_treebank_maxent_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_treebank_maxent_tagger.pkl',
            'nltk_treebank_ngram_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_treebank_ngram_tagger.pkl',
            'nltk_treebank_perceptron_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_treebank_perceptron_tagger.pkl',
            'nltk_treebank_tnt_tagger': 'https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/nltk_treebank_tnt_tagger.pkl'}

        if not MANIFEST:
            MANIFEST.update(DEFAULT_MANIFEST)


populate_manifest()
