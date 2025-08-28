class BootstrapFewShotWithRandomSearch(Teleprompter):
    """
    An experimental teleprompter that bootstraps multiple candidate sets of demonstrations and selects the best DSPy program based on evaluation scores.

    This teleprompter samples up to `num_candidate_programs` candidate demonstration sets by resampling and shuffling the training data, optionally bootstrapping further demonstrations. It then evaluates each candidate program using the provided `metric` on a validation set, tracks scores (with support for error limits), and returns the best-performing program.

    Args:
        metric (Callable): Evaluation metric used to score candidate programs.
        teacher_settings (dict): Settings for the teacher module, passed to underlying BootstrapFewShot.
        max_bootstrapped_demos (int): Maximum number of bootstrapped demonstrations per predictor.
        max_labeled_demos (int): Maximum labeled demonstrations from the training set per predictor.
        max_rounds (int): Maximum rounds of bootstrapping (per candidate program).
        num_candidate_programs (int): Number of candidate sets/programs to generate and evaluate.
        num_threads (int): Number of threads to use for parallelized scoring (optional).
        max_errors (int): Maximum allowed scoring errors before halting (defaults to DSPy global max_errors).
        stop_at_score (float): Early stopping when a candidate achieves this score or higher.
        metric_threshold (float): Optionally require metric >= threshold to accept a bootstrapped demo.

    Example:
        ```python
        from dspy.teleprompt import BootstrapFewShotWithRandomSearch
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=custom_metric,
            num_candidate_programs=8,
            max_bootstrapped_demos=3,
            max_labeled_demos=4,
            num_threads=4,
            stop_at_score=0.95,
            max_errors=15
        )
        optimized_program = teleprompter.compile(student=qa_module, trainset=train_examples)
        ```
    """
    def __init__(
        self,
        metric,
        teacher_settings=None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_candidate_programs=16,
        num_threads=None,
        max_errors=None,
        stop_at_score=None,
        metric_threshold=None,
    ):
        self.metric = metric
        self.teacher_settings = teacher_settings or {}
        self.max_rounds = max_rounds

        self.num_threads = num_threads
        self.stop_at_score = stop_at_score
        self.metric_threshold = metric_threshold
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.max_errors = max_errors
        self.num_candidate_sets = num_candidate_programs
        self.max_labeled_demos = max_labeled_demos

        print(f"Going to sample between {self.min_num_samples} and {self.max_num_samples} traces per predictor.")
        print(f"Will attempt to bootstrap {self.num_candidate_sets} candidate sets.")

    def compile(self, student, *, teacher=None, trainset, valset=None, restrict=None, labeled_sample=True):
        """
        Compile and optimize a DSPy program by bootstrapping and evaluating multiple candidate programs.

        Args:
            student (Module): The DSPy program to optimize.
            teacher (Module): Teacher model for bootstrapping demos (optional; defaults to student).
            trainset (list): List of training examples.
            valset (list): Evaluation data for scoring candidates (defaults to trainset).
            restrict: Optionally restrict candidate seeds (advanced usage).
            labeled_sample (bool): If True, sample labeled demos from trainset.

        Returns:
            Module: The best performing compiled program with candidate history.
        """
        self.trainset = trainset
        self.valset = valset or trainset

        effective_max_errors = self.max_errors if self.max_errors is not None else dspy.settings.max_errors

        scores = []
        all_subscores = []
        score_data = []

        for seed in range(-3, self.num_candidate_sets):
            if (restrict is not None) and (seed not in restrict):
                continue

            trainset_copy = list(self.trainset)

            if seed == -3:
                # zero-shot
                program = student.reset_copy()

            elif seed == -2:
                # labels only
                teleprompter = LabeledFewShot(k=self.max_labeled_demos)
                program = teleprompter.compile(student, trainset=trainset_copy, sample=labeled_sample)

            elif seed == -1:
                # unshuffled few-shot
                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=self.max_num_samples,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                    max_errors=effective_max_errors,
                )
                program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

            else:
                assert seed >= 0, seed

                random.Random(seed).shuffle(trainset_copy)
                size = random.Random(seed).randint(self.min_num_samples, self.max_num_samples)

                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=size,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                    max_errors=effective_max_errors,
                )

                program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

            evaluate = Evaluate(
                devset=self.valset,
                metric=self.metric,
                num_threads=self.num_threads,
                max_errors=effective_max_errors,
                display_table=False,
                display_progress=True,
            )

            result = evaluate(program)
            score, subscores = result.score, [output[2] for output in result.results]
            all_subscores.append(subscores)

            if len(scores) == 0 or score > max(scores):
                print("New best score:", score, "for seed", seed)
                best_program = program

            scores.append(score)
            print(f"Scores so far: {scores}")
            print(f"Best score so far: {max(scores)}")

            score_data.append({"score": score, "subscores": subscores, "seed": seed, "program": program})

            if self.stop_at_score is not None and score >= self.stop_at_score:
                print(f"Stopping early because score {score} is >= stop_at_score {self.stop_at_score}")
                break

        # Attach all evaluated candidate programs to the best performing program, sorted by score
        best_program.candidate_programs = score_data
        best_program.candidate_programs = sorted(
            best_program.candidate_programs, key=lambda x: x["score"], reverse=True
        )

        print(f"{len(best_program.candidate_programs)} candidate programs found.")
        return best_program
