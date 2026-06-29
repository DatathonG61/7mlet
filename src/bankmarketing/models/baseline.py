class BaselineModel:

    def recommend(self, customer):
        return {
            "offer": "CARTAO_CREDITO",
            "reason": "Oferta mais popular"
        }