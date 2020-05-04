from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import random
import math

# Change below BEFOREEEEE HAND it In
real_headlines_file = "clean_real.txt"
fake_headlines_file = "clean_fake.txt"


def load_data(vectorizer):
    # Load real headlines from file
    real_fp = open(real_headlines_file, "r")
    real_headlines = real_fp.readlines()
    real_fp.close()
    # Load fake headlines from file
    fake_fp = open(fake_headlines_file, "r")
    fake_headlines = fake_fp.readlines()
    fake_fp.close()
    #Label data
    labelled_headlines = []
    for line in real_headlines:
        labelled_headlines.append((line.strip(), "real"))
    for line in fake_headlines:
        labelled_headlines.append((line.strip(), "fake"))
    random.shuffle(labelled_headlines)
    

    # Initialize vectorizer
    headlines = vectorizer.fit_transform(headline for (headline, label) in labelled_headlines)
    labels = [label for (headline, label) in labelled_headlines]
    # Split up the data set
    num_headlines = len(labelled_headlines)
    num_test_valid = int(num_headlines * 0.15) 
    h_train_valid, h_test, l_train_valid, l_test = train_test_split(headlines, labels, test_size=num_test_valid,stratify=labels)
    h_train, h_valid, l_train, l_valid = train_test_split(h_train_valid, l_train_valid, test_size=num_test_valid,stratify=l_train_valid)
    return (h_train, l_train), (h_valid, l_valid), (h_test, l_test), labelled_headlines

def select_tree_model(train_data, valid_data):
    best_model = None 
    best_score = 0
    
    for criterion in ["gini", "entropy"]:
        for depth in [15, 30, 45, 60, 75]:
            model = DecisionTreeClassifier(criterion=criterion, max_depth=depth)
            model.fit(train_data[0], train_data[1])
            score = evaluate_training_model(model, valid_data)
            print("Tree with depth " + str(depth) + " and criterion " + str(criterion) + " : validation score = " + str(score))
            if score > best_score:
                best_score = score
                best_model = model
    return best_model, best_model.criterion, best_model.max_depth, best_score

def evaluate_training_model(training_model, valid_data):
    model_predicts = training_model.predict(valid_data[0])
    correct_count = 0
    total_headlines = len(valid_data[1])
    for i in range(total_headlines):
        if model_predicts[i] == valid_data[1][i]:
            correct_count += 1
    score = correct_count / total_headlines
    return score

def compute_information_gain(labelled_headlines, word):
    # split the data set base on if sentence contains the word
    num_total = len(labelled_headlines)
    if num_total == 0:
        return
    num_total_real, num_total_fake = 0, 0
    word_exist = [] # This is the right split
    word_no_exist = [] # This is the left split
    for data in labelled_headlines:
        headline, label = data[0], data[1]
        if word in headline:
            word_exist.append(data)
        else:
            word_no_exist.append(data)
        if label == "real":
            num_total_real += 1
    num_right = len(word_exist)
    num_left = len(word_no_exist)
    num_total_fake = num_total - num_total_real

    # find num real/fake on right split
    num_right_real = 0
    for data in word_exist:
        headline, label = data[0], data[1]
        if label == "real":
            num_right_real += 1
    num_right_fake = num_right - num_right_real

    # find num real/fake on left split
    num_left_real = 0
    for data in word_no_exist:
        headline, label = data[0], data[1]
        if label == "real":
            num_left_real += 1
    num_left_fake = num_left - num_left_real

    # Calculate H(Y)
    p_fake = num_total_fake / num_total
    p_real = num_total_real / num_total
    entropy_Y = - p_fake * math.log(p_fake, 2) - p_real * math.log(p_real, 2)

    # Calculate H(Y|x)
    p_left = (num_left / num_total)
    if num_left == 0:
        entropy_left = 0
    else:
        p_left_real = num_left_real / num_left
        p_left_fake = num_left_fake / num_left
        if p_left_real == 0:
            real_log_val = 0
        else:
            real_log_val = math.log(p_left_real, 2)
        if p_left_fake == 0:
            fake_log_val = 0
        else:
            fake_log_val = math.log(p_left_fake, 2)
        entropy_left = - p_left_fake * fake_log_val - p_left_real * real_log_val
    
    p_right = (num_right / num_total)
    if num_right == 0:
        entropy_right = 0
    else:
        p_right_real = num_right_real / num_right
        p_right_fake = num_right_fake / num_right
        if p_right_real == 0:
            real_log_val = 0
        else:
            real_log_val = math.log(p_right_real, 2)
        if p_right_fake == 0:
            fake_log_val = 0
        else:
            fake_log_val = math.log(p_right_fake, 2)
        entropy_right = - p_right_fake * fake_log_val - p_right_real * real_log_val
    
    entropy_Yx = (entropy_left * p_left) + (entropy_right * p_right)
    
    # Return IG(Y,x)
    return entropy_Y - entropy_Yx

def select_knn_model(train_data, valid_data):
    best_model = None 
    best_err = 1
    result = []
    train_result = []
    best_k = -1
    for k in range(1, 20):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(train_data[0], train_data[1])
        model.predict(valid_data[0])
        score = evaluate_training_model(model, valid_data)
        train_score = evaluate_training_model(model, train_data)
        result.append((k, (1 - score)))
        train_result.append((k, (1 - train_score)))
        if 1 - score < best_err:
            best_err = 1 - score
            best_k = k
            best_model = model
    return best_model, best_k, best_err, result, train_result

if __name__ == "__main__":
    vectorizer = CountVectorizer()
    train_data, valid_data, test_data, labelled_headlines = load_data(vectorizer)

    # Find best tree model
    best_tree_model = select_tree_model(train_data, valid_data)
    print("Best Tree Model is given by criterion: " + best_tree_model[1] + ", max depth: " + str(best_tree_model[2]) + ", validation score: " + str(best_tree_model[3]))

    # Output a dot file for visualization 
    tree.export_graphviz(best_tree_model[0], max_depth=2, out_file="Best_Tree.dot", class_names=best_tree_model[0].classes_, feature_names=vectorizer.get_feature_names())

    # Calculate Information gain
    print("Computed Information gain from the word 'the': " + str(compute_information_gain(labelled_headlines, "the")))
    print("Computed Information gain from the word 'hillary': " + str(compute_information_gain(labelled_headlines, "hillary")))
    print("Computed Information gain from the word 'donald': " + str(compute_information_gain(labelled_headlines, "donald")))
    print("Computed Information gain from the word 'trump': " + str(compute_information_gain(labelled_headlines, "Trump")))
    print("Computed Information gain from the word 'war': " + str(compute_information_gain(labelled_headlines, "war")))

    # Choose a knn model
    best_knn_model = select_knn_model(train_data, valid_data)
    print("Best kNN Model is given by k: " + str(best_knn_model[1]) + ", validation score: " + str(best_knn_model[2]))

    # Plotting
    x_axis = []
    y_axis = []
    x_train_axis = []
    y_train_axis = []
    for item in best_knn_model[3]:
        x_axis.append(item[0])
        y_axis.append(item[1])
    for item in best_knn_model[4]:
        x_train_axis.append(item[0])
        y_train_axis.append(item[1])
    plt.plot(x_axis, y_axis, x_train_axis, y_train_axis)
    plt.xticks(x_axis)
    plt.gca().invert_xaxis()
    plt.ylabel("test error")
    plt.legend(('Validation', 'Train'), loc='lower left')
    plt.show()
