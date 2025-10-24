- `"audio-classification"`: will return a [`AudioClassificationPipeline`].
- `"automatic-speech-recognition"`: will return a [`AutomaticSpeechRecognitionPipeline`].
- `"depth-estimation"`: will return a [`DepthEstimationPipeline`].
- `"document-question-answering"`: will return a [`DocumentQuestionAnsweringPipeline`].
- `"feature-extraction"`: will return a [`FeatureExtractionPipeline`].
- `"fill-mask"`: will return a [`FillMaskPipeline`]:.
- `"image-classification"`: will return a [`ImageClassificationPipeline`].
- `"image-feature-extraction"`: will return an [`ImageFeatureExtractionPipeline`].
- `"image-segmentation"`: will return a [`ImageSegmentationPipeline`].
- `"image-text-to-text"`: will return a [`ImageTextToTextPipeline`].
- `"image-to-image"`: will return a [`ImageToImagePipeline`].
- `"image-to-text"`: will return a [`ImageToTextPipeline`].
- `"mask-generation"`: will return a [`MaskGenerationPipeline`].
- `"object-detection"`: will return a [`ObjectDetectionPipeline`].
- `"question-answering"`: will return a [`QuestionAnsweringPipeline`].
- `"summarization"`: will return a [`SummarizationPipeline`].
- `"table-question-answering"`: will return a [`TableQuestionAnsweringPipeline`].
- `"text2text-generation"`: will return a [`Text2TextGenerationPipeline`].
- `"text-classification"` (alias `"sentiment-analysis"` available): will return a [`TextClassificationPipeline`].
- `"text-generation"`: will return a [`TextGenerationPipeline`]:.
- `"text-to-audio"` (alias `"text-to-speech"` available): will return a [`TextToAudioPipeline`]:.
- `"token-classification"` (alias `"ner"` available): will return a [`TokenClassificationPipeline`].
- `"translation"`: will return a [`TranslationPipeline`].
- `"translation_xx_to_yy"`: will return a [`TranslationPipeline`].
- `"video-classification"`: will return a [`VideoClassificationPipeline`].
- `"visual-question-answering"`: will return a [`VisualQuestionAnsweringPipeline`].
- `"zero-shot-classification"`: will return a [`ZeroShotClassificationPipeline`].
- `"zero-shot-image-classification"`: will return a [`ZeroShotImageClassificationPipeline`].
- `"zero-shot-audio-classification"`: will return a [`ZeroShotAudioClassificationPipeline`].
- `"zero-shot-object-detection"`: will return a [`ZeroShotObjectDetectionPipeline`].


Pipeline(_ScikitCompat, PushToHubMixin) (transformers.pipelines.base)
    TextToAudioPipeline(Pipeline) (transformers.pipelines.text_to_audio)
    ImageSegmentationPipeline(Pipeline) (transformers.pipelines.image_segmentation)
    Text2TextGenerationPipeline(Pipeline) (transformers.pipelines.text2text_generation)
        SummarizationPipeline(Text2TextGenerationPipeline) (transformers.pipelines.text2text_generation)
        TranslationPipeline(Text2TextGenerationPipeline) (transformers.pipelines.text2text_generation)
    ObjectDetectionPipeline(Pipeline) (transformers.pipelines.object_detection)
    ImageToTextPipeline(Pipeline) (transformers.pipelines.image_to_text)
    VisualQuestionAnsweringPipeline(Pipeline) (transformers.pipelines.visual_question_answering)
    FillMaskPipeline(Pipeline) (transformers.pipelines.fill_mask)
    VideoClassificationPipeline(Pipeline) (transformers.pipelines.video_classification)
    ImageToImagePipeline(Pipeline) (transformers.pipelines.image_to_image)
    ImageClassificationPipeline(Pipeline) (transformers.pipelines.image_classification)
    ChunkPipeline(Pipeline) (transformers.pipelines.base)
        TokenClassificationPipeline(ChunkPipeline) (transformers.pipelines.token_classification)
        QuestionAnsweringPipeline(ChunkPipeline) (transformers.pipelines.question_answering)
        MaskGenerationPipeline(ChunkPipeline) (transformers.pipelines.mask_generation)
        AutomaticSpeechRecognitionPipeline(ChunkPipeline) (transformers.pipelines.automatic_speech_recognition)
        ZeroShotObjectDetectionPipeline(ChunkPipeline) (transformers.pipelines.zero_shot_object_detection)
        ZeroShotClassificationPipeline(ChunkPipeline) (transformers.pipelines.zero_shot_classification)
        DocumentQuestionAnsweringPipeline(ChunkPipeline) (transformers.pipelines.document_question_answering)
    TextClassificationPipeline(Pipeline) (transformers.pipelines.text_classification)
    ZeroShotImageClassificationPipeline(Pipeline) (transformers.pipelines.zero_shot_image_classification)
    ImageTextToTextPipeline(Pipeline) (transformers.pipelines.image_text_to_text)
    ImageFeatureExtractionPipeline(Pipeline) (transformers.pipelines.image_feature_extraction)
    AudioClassificationPipeline(Pipeline) (transformers.pipelines.audio_classification)
    DepthEstimationPipeline(Pipeline) (transformers.pipelines.depth_estimation)
    TableQuestionAnsweringPipeline(Pipeline) (transformers.pipelines.table_question_answering)
    ZeroShotAudioClassificationPipeline(Pipeline) (transformers.pipelines.zero_shot_audio_classification)
    TextGenerationPipeline(Pipeline) (transformers.pipelines.text_generation)
        H2OTextGenerationPipeline(TextGenerationPipeline) (h2oai_pipeline)
    FeatureExtractionPipeline(Pipeline) (transformers.pipelines.feature_extraction)
