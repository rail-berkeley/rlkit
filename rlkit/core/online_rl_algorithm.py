import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm

class OnlineRLAlgorithm(BaseRLAlgorithm):
    def _train(self):
        self.expl_data_collector.start_collection()
        for i in enumerate(range(self.min_num_steps_before_training)):
            self.expl_data_collector.collect_new_step(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
        init_expl_paths = self.expl_data_collector.end_collection()
        self.replay_buffer.add_paths(init_expl_paths)
        gt.stamp('initial exploration', unique=True)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            self.expl_data_collector.start_collection()
            for _ in range(self.num_trains_per_train_loop):
                self.expl_data_collector.collect_new_step(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                )
                gt.stamp('exploration sampling', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            new_expl_paths = self.expl_data_collector.end_collection()
            self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)

            self._end_epoch(epoch)