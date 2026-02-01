from rest_framework import serializers
from .models import AnalysisSession, Visualization, MLModel, HypothesisTest


class AnalysisSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisSession
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at', 'status',
                           'result', 'code_generated', 'error_message',
                           'celery_task_id', 'execution_time']


class AnalysisSessionListSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisSession
        fields = ['id', 'name', 'analysis_type', 'status', 'dataset',
                 'execution_time', 'created_at']


class RunEDASerializer(serializers.Serializer):
    dataset_id = serializers.UUIDField()
    query = serializers.CharField(help_text='Natural language EDA query')
    name = serializers.CharField(max_length=255, required=False, default='EDA Analysis')


class RunVisualizationSerializer(serializers.Serializer):
    dataset_id = serializers.UUIDField()
    query = serializers.CharField(help_text='Natural language visualization request')
    chart_type = serializers.ChoiceField(
        choices=['auto', 'kde', 'histogram', 'scatter', 'bar', 'line',
                 'heatmap', 'box', 'violin', 'pie', 'pair'],
        default='auto',
    )
    name = serializers.CharField(max_length=255, required=False, default='Visualization')


class RunMLModelSerializer(serializers.Serializer):
    dataset_id = serializers.UUIDField()
    query = serializers.CharField(help_text='Natural language ML request')
    model_type = serializers.ChoiceField(
        choices=['auto', 'logistic_regression', 'random_forest', 'xgboost',
                 'svm', 'neural_network', 'decision_tree', 'knn',
                 'gradient_boosting', 'linear_regression'],
        default='auto',
    )
    target_column = serializers.CharField(required=False, default='')
    name = serializers.CharField(max_length=255, required=False, default='ML Model')


class RunDLModelSerializer(serializers.Serializer):
    dataset_id = serializers.UUIDField()
    query = serializers.CharField(help_text='Natural language DL request')
    model_type = serializers.ChoiceField(
        choices=['auto', 'cnn', 'rnn', 'lstm', 'gru', 'transformer',
                 'autoencoder', 'gan', 'mlp', 'resnet'],
        default='auto',
    )
    framework = serializers.ChoiceField(
        choices=['auto', 'pytorch', 'tensorflow'],
        default='pytorch',
    )
    target_column = serializers.CharField(required=False, default='')
    epochs = serializers.IntegerField(required=False, default=50, min_value=1, max_value=1000)
    batch_size = serializers.IntegerField(required=False, default=32, min_value=1, max_value=4096)
    learning_rate = serializers.FloatField(required=False, default=0.001)
    task_type = serializers.ChoiceField(
        choices=['auto', 'classification', 'regression', 'sequence_prediction',
                 'anomaly_detection', 'generative'],
        default='auto',
    )
    name = serializers.CharField(max_length=255, required=False, default='DL Model')


class RunHypothesisTestSerializer(serializers.Serializer):
    dataset_id = serializers.UUIDField()
    query = serializers.CharField(help_text='Natural language hypothesis test request')
    test_type = serializers.ChoiceField(
        choices=['auto', 't_test', 'chi_square', 'anova', 'mann_whitney',
                 'pearson', 'spearman', 'shapiro'],
        default='auto',
    )
    name = serializers.CharField(max_length=255, required=False, default='Hypothesis Test')


class RunSQLQuerySerializer(serializers.Serializer):
    dataset_id = serializers.UUIDField(required=False)
    connection_id = serializers.UUIDField(required=False)
    query = serializers.CharField(help_text='Natural language SQL query')
    name = serializers.CharField(max_length=255, required=False, default='SQL Query')


class VisualizationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Visualization
        fields = '__all__'
        read_only_fields = ['id', 'created_at']


class MLModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModel
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at']


class MLModelListSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModel
        fields = ['id', 'name', 'model_type', 'task_type', 'framework', 'metrics',
                 'target_column', 'epochs', 'created_at']


class HypothesisTestSerializer(serializers.ModelSerializer):
    class Meta:
        model = HypothesisTest
        fields = '__all__'
        read_only_fields = ['id', 'created_at']
