from models.autobot.autobot import AutoBotEgo
from models.wayformer.wayformer import Wayformer
from models.mtr.MTR import MotionTransformer

__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
}


def build_model(config):

    model = __all__[config.method.model_name](
        config=config
    )

    return model
