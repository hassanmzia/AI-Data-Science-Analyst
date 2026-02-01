from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='analysissession',
            name='analysis_type',
            field=models.CharField(
                choices=[
                    ('eda', 'Exploratory Data Analysis'),
                    ('visualization', 'Data Visualization'),
                    ('hypothesis', 'Hypothesis Testing'),
                    ('ml_model', 'Machine Learning Model'),
                    ('dl_model', 'Deep Learning Model'),
                    ('sql_query', 'SQL Query'),
                    ('general', 'General Analysis'),
                ],
                max_length=20,
            ),
        ),
        migrations.AlterField(
            model_name='mlmodel',
            name='model_type',
            field=models.CharField(
                choices=[
                    ('logistic_regression', 'Logistic Regression'),
                    ('random_forest', 'Random Forest'),
                    ('xgboost', 'XGBoost'),
                    ('svm', 'Support Vector Machine'),
                    ('neural_network', 'Neural Network'),
                    ('decision_tree', 'Decision Tree'),
                    ('knn', 'K-Nearest Neighbors'),
                    ('gradient_boosting', 'Gradient Boosting'),
                    ('linear_regression', 'Linear Regression'),
                    ('cnn', 'Convolutional Neural Network'),
                    ('rnn', 'Recurrent Neural Network'),
                    ('lstm', 'Long Short-Term Memory'),
                    ('gru', 'Gated Recurrent Unit'),
                    ('transformer', 'Transformer'),
                    ('autoencoder', 'Autoencoder'),
                    ('gan', 'Generative Adversarial Network'),
                    ('mlp', 'Multi-Layer Perceptron'),
                    ('resnet', 'ResNet'),
                    ('custom', 'Custom Model'),
                ],
                max_length=30,
            ),
        ),
        migrations.AlterField(
            model_name='mlmodel',
            name='task_type',
            field=models.CharField(
                choices=[
                    ('classification', 'Classification'),
                    ('regression', 'Regression'),
                    ('clustering', 'Clustering'),
                    ('image_classification', 'Image Classification'),
                    ('text_classification', 'Text Classification'),
                    ('sequence_prediction', 'Sequence Prediction'),
                    ('anomaly_detection', 'Anomaly Detection'),
                    ('generative', 'Generative'),
                ],
                default='classification',
                max_length=30,
            ),
        ),
        migrations.AddField(
            model_name='mlmodel',
            name='framework',
            field=models.CharField(
                choices=[
                    ('sklearn', 'Scikit-learn'),
                    ('pytorch', 'PyTorch'),
                    ('tensorflow', 'TensorFlow'),
                    ('auto', 'Auto Select'),
                ],
                default='sklearn',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='mlmodel',
            name='epochs',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='mlmodel',
            name='batch_size',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='mlmodel',
            name='learning_rate',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
