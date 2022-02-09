class Card():
    def __init__(self):
        # self.all_num = 54
        # self.card_num = {"JOKER":2,"2":4,"A":4,"K":4,"Q":4,"J":4,"10":4,"9":4,"8":4,"7":4,"6":4,"5":4,"4":4,"3":4}
        # self.player_a_all = [] # 上家出过的牌
        # self.player_b_all = [] # 我出过的牌
        # self.player_c_all = [] # 下家出过的牌
        self.player_a_last = [] # 上家上次出牌
        self.player_b_last = [] # 我上次出牌
        self.player_c_last = [] # 下家上次出牌

    def update(self, pos, mask, cards):
        a_card=[]
        b_card=[]
        c_card=[]
        for i,c in enumerate(pos):
            if mask[i]==0:
                #0横，1竖
                if c[1] < 365 and c[1] > 270:
                    if c[0] < 520:
                        a_card.append(cards[i])
                    else:
                        c_card.append(cards[i])
                elif c[1] > 365 and c[1] < 520:
                    b_card.append(cards[i])

        if a_card:
            a_card.sort()
            if a_card != self.player_a_last:
                print("上家",a_card)
                self.player_a_last = a_card
            else:
                a_card = []
        else:
            self.player_a_last = []
        if b_card:
            b_card.sort()
            if b_card != self.player_b_last:
                print("我",b_card)
                self.player_b_last = b_card
            else:
                b_card = []
        else:
            self.player_b_last = []
        if c_card:
            c_card.sort()
            if c_card != self.player_c_last:
                print("下家",c_card)
                self.player_c_last = c_card
            else:
                c_card = []
        else:
            self.player_c_last = []

        return a_card, b_card, c_card