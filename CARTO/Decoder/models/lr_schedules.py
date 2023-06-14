# from typing import List

from CARTO.Decoder.config import LearningRateScheduleConfig, LearningRateScheduleType


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass

    @staticmethod
    def get_from_config(cfg: LearningRateScheduleConfig):
        if cfg.type == LearningRateScheduleType.STEP:
            return StepLearningRateSchedule(
                cfg.initial,
                cfg.interval,
                cfg.factor,
            )
        elif cfg.type == LearningRateScheduleType.WARMUP:
            return WarmupLearningRateSchedule(
                cfg.initial,
                cfg.final,
                cfg.length,
            )

        elif cfg.type == LearningRateScheduleType.CONSTANT:
            return ConstantLearningRateSchedule(cfg.initial)
        elif cfg.type == LearningRateScheduleType.LEVEL_DECAY:
            return LevelDecayLearningRateSchedule(cfg.initial, cfg.factor)
        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(cfg.type)
            )


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


class LevelDecayLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, decay):
        self.initial = initial
        self.decay = decay
        self.level = 0

    def inc_level(self, level=1):
        self.level += level

    def get_learning_rate(self, epoch):
        """
        Epoch does not matter
        """
        return self.initial * ((self.decay) ** self.level)


# def get_learning_rate_schedules(schedulers: List[LearningRateSchedulerConfig]):
#   schedules = []
#   for schedule in schedulers:

#   return schedules
