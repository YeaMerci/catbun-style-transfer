from processing import ImageProcessing
import argparse


def main(style_url: str, content_url: str):
    process = ImageProcessing()
    output = process(content_url=style_url, style_url=content_url)
    process.show(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Specify two arguments: "
                    "1) style - the path to the image whose style will be transferred. "
                    "2) content - the path to the image to which the style will be transferred."
    )
    parser.add_argument(
        "style", type=str,
        help="style - the path to the image whose style will be transferred."
    )

    parser.add_argument(
        "content", type=str,
        help="content - the path to the image to which the style will be transferred."
    )

    args = parser.parse_args()
    main(style_url=args.style, content_url=args.content)
