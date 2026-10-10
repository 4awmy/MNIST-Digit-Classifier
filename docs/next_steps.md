# Next steps after the local implementation

The public repository is [4awmy/MNIST-Digit-Classifier](https://github.com/4awmy/MNIST-Digit-Classifier), and the team has sent the friend invitation. Confirm acceptance in GitHub if needed.

1. **Review the finished local evidence.** Read [report](report.md), [methodology](methodology.md) and [change explanation](changes.md). Study the selected settings, validation-based choice and error examples so the team can explain them. Keep the three notebooks separate; they supply three saved models to one app.
2. **Deploy yourself after the GitHub update.** In Streamlit Community Cloud select `4awmy/MNIST-Digit-Classifier`, branch `main`, entry point `app.py`, Python 3.12. Follow [deployment](deployment.md). Deployment loads the committed artifacts; it does not retrain.
3. **Verify the live deployment.** Check blank input, drawing, Predict, Clear, all three models, touch interaction and the processed-image preview. Inspect cold start, memory and response times on the actual cloud instance. Add the real URL to README/report only after this succeeds.
4. **Finish the team review.** Confirm the assignment deliverables, report conclusions and collaborator access. Use separate branches/pull requests for subsequent edits. No names have been added to the report, as requested.
5. **Add picture upload afterward.** Start with one clear digit in a PNG/JPEG. Handle grayscale/polarity and feed it through the existing preprocessing, with a preview and blank/unreadable-image validation. Test representative uploaded images before claiming accuracy. Arbitrary photos or multi-digit numbers need additional segmentation work.

There is no need to repeat training for a GitHub push, documentation edit or app deployment. See [changes](changes.md) for the cases that do require retraining.
