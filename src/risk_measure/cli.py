import argparse
import logging
from pathlib import Path
import importlib.resources as resources

from . import (
    load_glove,
    expand_vocabulary,
    load_word_list,
    save_word_list,
    score_transcripts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the ``run-measure`` console script."""
    parser = argparse.ArgumentParser(
        description="Measure supply chain risk from transcripts"
    )
    parser.add_argument("excel", type=Path, help="Path to transcript Excel file")
    parser.add_argument(
        "--sc-words",
        type=Path,
        default=None,
        help="Supply chain word list",
    )
    parser.add_argument(
        "--risk-words",
        type=Path,
        default=None,
        help="Risk word list",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scores.xlsx"),
        help="Output Excel path",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Expand word lists using GloVe",
    )
    args = parser.parse_args()

    sc_path = (
        args.sc_words
        if args.sc_words is not None
        else resources.files("risk_measure").joinpath("data/expanded_sc_words.txt")
    )
    risk_path = (
        args.risk_words
        if args.risk_words is not None
        else resources.files("risk_measure").joinpath("data/expanded_risk_words.txt")
    )

    if args.expand:
        embeddings = load_glove()
        sc_initial = load_word_list(sc_path)
        risk_initial = load_word_list(risk_path)
        sc_exp = expand_vocabulary(sc_initial, embeddings)
        risk_exp = expand_vocabulary(risk_initial, embeddings)
        if args.sc_words is not None:
            save_word_list(sc_exp, sc_path)
        if args.risk_words is not None:
            save_word_list(risk_exp, risk_path)

    sc_words = load_word_list(sc_path)
    risk_words = load_word_list(risk_path)
    df = score_transcripts(args.excel, sc_words, risk_words)
    df.to_excel(args.output, index=False)
    logger.info("Scores saved to %s", args.output)


if __name__ == "__main__":
    main()
