from uaf_gravity.evolution import UAFGravityEvolution
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("="*70)
    print("🚀 XGrav — UAF-Powered Evolutionary Gravity Laboratory")
    print("="*70)

    evo = UAFGravityEvolution(
        n_asteroids=124,   # + 5 shepherds + 1 central = 130 тел
        pop_size=10,
        generations=20
    )

    print("Запуск эволюции...\n")
    best_config, best_score, history = evo.evolve()

    print(f"\n✅ Лучшая конфигурация найдена!")
    print(f"UAF Stability Score: {best_score:.4f}")
    
    # Простой график истории
    plt.figure(figsize=(10, 5))
    plt.plot(history, marker='o', linestyle='-', color='blue')
    plt.title('Эволюция UAF Stability Score')
    plt.xlabel('Поколение')
    plt.ylabel('Stability Score')
    plt.grid(True)
    plt.show()
