# Streamlit Community Cloud

Repository: [4awmy/MNIST-Digit-Classifier](https://github.com/4awmy/MNIST-Digit-Classifier).

The user will perform the Cloud deployment after the local training and GitHub update. No live deployment is claimed here. Pushing the repository and deploying the app do not retrain the models; see [training changes](../README.md#training-changes).

The deployment entry point is `app.py`, branch `main`, Python **3.12**. `requirements.txt` contains application dependencies; the larger GPU/training requirements are separate. All three final artifacts in `models/` must be present in GitHub. The app runs inference on CPU and does not train or download models on startup.

Streamlit Community Cloud does not provide GPU computation ([Streamlit confirmation](https://discuss.streamlit.io/t/streamlit-app-is-crashing-during-inference/66566/2)). Training the networks on a GPU does not prevent CPU inference: the saved files contain their architecture and learned weights. The application uses `tensorflow-cpu`, and loading those files on CPU does not require retraining. CUDA requirements belong only to the local training environment.

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using the GitHub account with access to this repository.
2. Choose **Create app**, then deploy an existing GitHub repository.
3. Select `4awmy/MNIST-Digit-Classifier`, branch `main`, entry point `app.py`.
4. Open advanced settings and select Python 3.12. No application secrets are required.
5. Deploy and inspect installation/startup logs. Record the actual public app URL in README and the report.
6. Test blank input, drawing, Predict, Clear, all three model choices, touch interaction, and the processed-image preview on the deployed app.

Cloud deployment is complete only after the live URL loads and all three saved models predict successfully. If installation or model loading fails, compare the pinned package versions with model metadata before changing dependencies. The RF artifact uses compression to limit repository size; its memory use after loading exceeds its compressed size. Keep model objects cached, and measure live startup and response times rather than treating local timing as cloud timing.

Official instructions: [deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) and [app dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
