import type { ModelName } from "../lib/ws";
import { Seg } from "./ModeSwitch";

type Props = {
  model: ModelName;
  onChange: (m: ModelName) => void;
  disabled?: boolean;
};

export function ModelSwitch({ model, onChange, disabled }: Props) {
  return (
    <div className={disabled ? "opacity-40 pointer-events-none" : ""}>
      <Seg
        ariaLabel="Alphabet model"
        value={model}
        onChange={onChange}
        options={[
          { id: "cnn", label: "CNN" },
          { id: "mobilenet", label: "MobileNet" },
        ]}
      />
    </div>
  );
}
