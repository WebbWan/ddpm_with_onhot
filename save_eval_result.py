from eval.show_eval_result import showMethodResult

if __name__ == '__main__':
    methods = ["LogisticRegression", "MLP", "SVM"]
    # for method in methods:
    #     showMethodResult(method, "./eval/eval_result/{}/".format(method), False)

    baslines = ["oversampling", "smote"]
    for basline in baslines:
        for method in methods:
            showMethodResult(method, "./eval/{}_result/{}/".format(basline, method), True, basline)
