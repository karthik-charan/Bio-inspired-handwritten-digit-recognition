# Splits the data from text file into classwise train and test data

line_no = 1
class_label = 0
path_label = "train"

path_val = 100
class_val = 200

filename = ""

with open('mfeat-pix.txt') as f:
    for line in f:
        curr = line.strip()

        filename = "./Data/"+str(path_label)+"_"+str(class_label)+".txt"
        with open(filename, 'a') as r:
            r.write(curr)
            r.write('\n')

        # Testing Verbose
        # print(curr)
        # print("line number: ", line_no)
        # print("class label: ", class_label)
        # print("path label: ", path_label)
        
        if line_no == path_val:
            path_val+=100
            if path_label == "train":
                path_label = "test"
            else:
                path_label = "train"

        if line_no == class_val:
            class_val += 200
            class_label += 1
        
        line_no += 1
        
        # For testing purposes
        # if class_label == 1:
        #     break