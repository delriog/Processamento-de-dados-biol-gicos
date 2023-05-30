from funcoes import *

TAXA_AMOSTRAGEM = 250
TEMPO_ATUALIZACAO = 1
TAMANHO_BUFFER = 5
ESCALA = False

def main() :
    arquivo = sys.argv[1]
    dataset = carregar_arquivo(arquivo)

    params = input("\nDigite os seguintes parâmetros separados por espaço:\n--------------------------------------------\nTaxa de amostragem\nTempo de atualização (segundos)... Obs: Float\nTamanho do Buffer (s)\nNormalização (s/n)\nSimulação (s/n)\n--------------------------------------------\n\n").split(" ")

    TAXA_AMOSTRAGEM = int(params[0])
    TEMPO_ATUALIZACAO = float(params[1])
    TAMANHO_BUFFER = float(params[2])
    if params[3] == "s": NORM = True
    else: NORM = False
    if params[4] == "s": SIMULACAO = True
    else: SIMULACAO = False

    X = dataset.swapaxes(1, 0)

    linhas = X.shape[1]
    # linhas = 2000
    intervalo = TAXA_AMOSTRAGEM * TEMPO_ATUALIZACAO
    intervalo_final = 0
    intervalo_final += intervalo
    buffer = TAXA_AMOSTRAGEM * TAMANHO_BUFFER
    features = list()

    dado = []
    
    if SIMULACAO: 
        print("Fazendo leitura do arquivo...\n")
        time.sleep(TAMANHO_BUFFER)

    for inicio in range(0, linhas-int(intervalo)+1, int(intervalo)):

        if SIMULACAO:
            time.sleep(TEMPO_ATUALIZACAO)

        X_f = dc(X[:, inicio:(int(intervalo_final)+int(buffer)-TAXA_AMOSTRAGEM)])

        for i in range(0, 10):
            X_f = butter_notch(X_f, 60)
            X_f = butter_notch(X_f, 120)
            X_f = butter_lowpass(X_f, 50.)
            X_f = butter_highpass(X_f, 3.)

        # Tarefa 1
        f, Pxx = welch(X_f)

        features = list()

        X_average = np.average(Pxx, axis=0)

        # Tarefa 2
        for mi, ma in [delta, theta, alpha, beta, gamma]:
            features.append(X_average[mi:ma])

        features = [np.average(f) for f in features]

        # Tarefa 4 
        if NORM:
            features = minmax_scale(features, feature_range=(0, 100))

        dado.append([inicio, int(intervalo_final), features[2], features[3], features[0], features[1], features[4]])

        if SIMULACAO: 
            os.system('clear')
            print("\nInicio: ", inicio, "Final: ", int(intervalo_final)+int(buffer)-TAXA_AMOSTRAGEM, '\nalpha: ', features[2], '\nbeta: ', features[3], '\ndelta: ', features[0], '\ntheta: ', features[1], '\ngamma: ', features[4], "\n")
        else:
            print("\nInicio: ", inicio, "Final: ", int(intervalo_final)+int(buffer)-TAXA_AMOSTRAGEM, '\nalpha: ', features[2], 'beta: ', features[3], 'delta: ', features[0], 'theta: ', features[1], 'gamma: ', features[4], "\n")

        intervalo_final += intervalo

    df = pd.DataFrame(dado, columns=['inicio', 'fim', 'alpha', 'beta', 'delta', 'theta', 'gamma'])
    df.to_csv(arquivo.split(".")[0]+ '.csv',index=False)



if __name__=="__main__":
    main()