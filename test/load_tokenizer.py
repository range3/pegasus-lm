from pprint import pprint
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
from pegasuslm import PegasusGPT2Config, PegasusGPT2Tokenizer


def main():
    model_path = Path(__file__).resolve().parent.parent / "model/50KV_200MS"

    # config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # print(tokenizer.normalize("  あい～うえお*１２３１\n  1\n a"))

    print(tokenizer.vocab_file)
    print(tokenizer.vocab_size)
    print(tokenizer.all_special_tokens)
    # print(tokenizer.get_vocab())
    # return

    sentences = [
        "こんにちは",
        "吾輩は猫である。名前はまだ無い",
        "ABCDaAaA",
        " 前後にスペース    ",
        "<s>[CLS]あれは何ですか？[SEP][MASK]は猫です。[SEP]</s>",
    ]
    pprint(tokenizer(sentences, padding=True))

    for s in sentences:
        print(tokenizer.tokenize(s))
        # print(tokenizer.sp_model.encode(s, out_type=str))

    # long sentences
    s = """
    ２４日夜の大雪の影響でＪＲ京都線と琵琶湖線は、一時、１５本の電車が乗客を乗せたまま駅の間で動けない状態になり、体調不良を訴える人も出ました。
    ＪＲ西日本では、京都線や琵琶湖線を含む多くの路線で運転の見合わせが続いています。

    ＪＲ西日本によりますと、大雪の影響で京都線は、京都府向日市にある向日町駅付近など複数の地点で線路のポイントが切り替わらなくなり、２４日午後８時ごろから京都・大阪間の全線で運転を見合わせています。
    京都と滋賀を結ぶ琵琶湖線でも全線で見合わせが続いていて、運転再開のめどは立っていません。
    この影響で、京都線と琵琶湖線はこれまでにあわせて１５本の電車が駅の間で乗客を乗せたまま動けない状態になり、少なくとも３本の電車で体調不良を訴える人が出たということです。
    このため、ＪＲは乗客を線路に降ろして駅まで歩いて移動してもらうなどの対応をとったということですが、帰宅する手段がないため駅に停車する電車の中や駅の連絡通路、市の施設で待機を余儀なくされる人が出ました。
    また、兵庫県内を走る山陽線でも電車が動けなくなり、乗客を線路に降ろしたということです。
    ＪＲ西日本では、午前１１時半現在、京都線や琵琶湖線、それに神戸線など関西を走る主要な路線の多くで運転を見合わせていて、交通網の混乱が続いています。
    """
    # print(tokenizer.normalize(s))
    t = tokenizer.encode(s)
    print(t)
    print(tokenizer.decode(t))

    # config = PegasusGPT2Config()
    # config.save_pretrained()


if __name__ == "__main__":
    main()
