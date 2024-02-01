#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importer la bibliothèque FANN (Fast Artificial Neural Network)
from fann2 import libfann
import os # wget linux command

def recupere_document(filename):
    #file without extension
    file_no_extension = filename.split('.')[0]
    file_no_extension = file_no_extension.split('pg')[-1]
    os.system('wget -O ' + filename + ' http://www.gutenberg.org/cache/epub/' + file_no_extension + '/' + filename)

def recupere_texte(book):
    with open(book) as f:
        file = f.read()
    return file


def preprocess(document):
    document = document.lower() #minuscule
    # Initialiser un tableau pour stocker les fréquences des lettres
    letter_frequencies = [0] * 26
    # Compter la fréquence de chaque lettre dans le document
    for letter in document:
        if 'a' <= letter <= 'z':
            index = ord(letter) - ord('a')
            letter_frequencies[index] += 1
    # Normaliser les fréquences des lettres pour qu'elles soient toutes entre 0 et 1
    total_letters = sum(letter_frequencies)
    letter_frequencies = [freq / total_letters for freq in letter_frequencies]

    return letter_frequencies


# Création du réseau de neurones
num_input = 26  # Nombre de lettres dans l'alphabet
num_hidden = 12  # Nombre de neurones cachés
num_output = 2  # Français ou Anglais dans ce cas

desired_error = 0.001
max_iterations = 100000
iterations_between_reports = 1000

ann = libfann.neural_net()
ann.create_standard_array([num_input, num_hidden, num_output])

ann.set_learning_rate(0.7)

# fonction sigmoïde : f(x) = 1 / (1 + exp(-x))
ann.set_activation_function_output(libfann.SIGMOID_STEPWISE)

# fichier de frequences elabore par un corpus de 12 textes en francais et anglais sur gutenberg.org/ebooks
# creer fichier frequences.data
liste_FR = ['pg17519.txt', 'pg41211.txt', 'pg71062.txt', 'pg14155.txt', 'pg6838.txt', 'pg5318.txt']
liste_EN = ['pg4060.txt', 'pg28516.txt', 'pg6599.txt', 'pg84.txt', 'pg174.txt', 'pg2536.txt']

with open('frequences.data', 'w') as f:
    f.write(f'{len(liste_FR) + len(liste_EN)} {num_input} {num_output}\n')
    for book in liste_FR:
        if not os.path.exists(book):
            recupere_document(book)
        text = recupere_texte(book)
        prep = preprocess(text)
        f.write(' '.join(str(x) for x in prep) + '\n0 1\n')
    for book in liste_EN:
        if not os.path.exists(book):
            recupere_document(book)
        text = recupere_texte(book)
        prep = preprocess(text)
        f.write(' '.join(str(x) for x in prep) + '\n1 0\n')

ann.train_on_file('frequences.data', max_iterations, iterations_between_reports, desired_error)

# Sauvegarde du réseau de neurones entraîné
ann.save('langue_classify.net')



# Charger le réseau de neurones entraîné
ann = libfann.neural_net()
ann.create_from_file('langue_classify.net')

# le(s) texte(s) a analyser
frnechbook = 'pg14158.txt' #de l'origine des especes, charles darwin
engbook = 'pg4300.txt'  #Ulysses by James Joyce

if not os.path.exists(frnechbook):
    recupere_document(frnechbook)
if not os.path.exists(engbook):
    recupere_document(engbook)

texte1 = recupere_texte(frnechbook)
texte2 = recupere_texte(engbook)

prep = preprocess(texte1)

# Obtenir les prédictions
output = ann.run(prep)

# Interpréter les prédictions
if output[0] > output[1]:
    print(f"Document {frnechbook}: ENG")
else:
    print(f"Document {frnechbook}: FR")


prep = preprocess(texte2)

# Obtenir les prédictions
output = ann.run(prep)

# Interpréter les prédictions
if output[0] > output[1]:
    print(f"Document {engbook}: ENG")
else:
    print(f"Document {engbook}: FR")




# Max epochs   100000. Desired error: 0.0010000000.
# Epochs            1. Current error: 0.2513779700. Bit fail 24.
# Epochs           22. Current error: 0.0007459051. Bit fail 0.
# Document pg14158.txt: FR
# Document pg4300.txt: ENG

